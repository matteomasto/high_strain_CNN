import argparse
import datetime
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import (
    Input,
    Conv3D,
    Conv3DTranspose,
    LeakyReLU,
    MaxPool3D,
    Concatenate,
    Dense,
    Reshape
)
from tensorflow.keras.models import Model

# --- Data Generator ---
def load_and_process_file(file_path, dim, input_log_data):
    """Load and process a single .npz file"""
    def _load_npz(path):
        data = np.load(path.numpy().decode('utf-8'))
        Ilin = data['I'].astype('float32')
        phi = data['phi'].astype('float32')
        
        center = tuple(np.array(dim)//2)
        phi = phi - phi[center]
        
        Ilog = np.log(Ilin+1) if input_log_data else Ilin
        
        amp_norm = np.interp(
            np.sqrt(Ilin),
            (np.sqrt(Ilin).min(), np.sqrt(Ilin).max()),
            (0, 1)
        ).astype('float32')
        
        I = np.interp(
            Ilog,
            (Ilog.min(), Ilog.max()),
            (0, 1)
        ).astype('float32')
        
        complex_field = np.exp(1.0j * phi).astype('complex64')
        
        X = I[..., np.newaxis]  
        y = np.stack([
            I * complex_field,
            amp_norm * complex_field
        ], axis=-1)
        
        return X, y
    
    return tf.py_function(
        _load_npz,
        [file_path],
        [tf.float32, tf.complex64]
    )

def create_dataset(file_paths, batch_size, dim, input_log_data=False, shuffle=True, repeat=True):
    """Create a tf.data.Dataset from file paths"""
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(file_paths))
    
    if repeat:
        dataset = dataset.repeat()
    
    dataset = dataset.map(
        lambda x: load_and_process_file(x, dim, input_log_data),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    dataset = dataset.map(lambda x, y: (
        tf.reshape(x, (*dim, 1)),
        tf.reshape(y, (*dim, 2))
    ))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
    
# --- Loss ---
def subfunc_WCA(mod_log, phi_true, phi_pred):  
    batch_size = tf.shape(phi_true)[0]
    glob_shift = tf.reshape(tf.math.reduce_mean((phi_true - phi_pred), axis = (1,2,3)), (batch_size,1,1,1))

    weights  = mod_log / tf.math.reduce_sum(mod_log, axis=(1,2,3), keepdims=True)
    error    = tf.cast(weights, 'complex64') * tf.math.exp(1.0j * (phi_true - phi_pred - glob_shift))

    loss      = 1 - tf.math.abs(tf.math.reduce_sum(error, axis=(1,2,3)))  
    return loss

def loss_combo(y_true, y_pred):
    phi_pred = tf.complex(y_pred[...,0], 0.0)
    phi_true = tf.complex(tf.math.angle(y_true[...,1]), 0.0)
    mod_log = tf.abs(y_true[...,0])
    
    err1 = subfunc_WCA(mod_log, phi_true, phi_pred)
    err2 = subfunc_WCA(mod_log, -phi_true, phi_pred)
    
    return tf.minimum(err1, err2)

# --- Model (UNet) ---
def encoder_block(x_input, num_filters, ker):

    x = Conv3D(num_filters, ker, strides=1, padding="same")(x_input)
    s = LeakyReLU(alpha=0.2)(x)
    x = MaxPool3D(pool_size=2)(s)

    return x, s
    
def encoder_block_mod(x_input, ker, num_filters, rate):

    reg = num_filters // 4

    x1 = Conv3D(reg, ker, strides=1, dilation_rate=rate[0], padding="same")(x_input)
    x2 = Conv3D(reg, ker, strides=1, dilation_rate=rate[1], padding="same")(x_input)
    x3 = Conv3D(reg, ker, strides=1, dilation_rate=rate[2], padding="same")(x_input)
    x4 = Conv3D(reg, ker, strides=1, dilation_rate=rate[3], padding="same")(x_input)
    x = tf.concat([x_input, x1, x2, x3, x4], axis=-1)

    s = LeakyReLU(alpha=0.2)(x)
    x = MaxPool3D(pool_size=2)(s)

    return x, s
    
def decoder_block(x_input, num_filters, ker, skip_input=None):

    if skip_input is not None:
        x_input = Concatenate()([x_input, skip_input])

    x = Conv3DTranspose(num_filters, ker, strides=2, padding="same")(x_input)
    x = LeakyReLU(alpha=0.2)(x)

    return x

def skip_block(x_input, ker):

    num_filters = x_input.shape[-1] // 2
    x = Conv3D(num_filters, ker, strides=1, padding="same")(x_input)
    x = LeakyReLU(alpha=0.2)(x)
    return x
    
def build_unet(input_shape):
    inputs = Input(input_shape)

    # ─── ENCODER ─────────────────────────────────────────
    x, s1 = encoder_block_mod(inputs, ker=5, num_filters=32,  rate=[8,5,3,1]) #32
    x, s2 = encoder_block_mod(x,      ker=5, num_filters=64, rate=[6,4,2,1]) #16
    x, s3 = encoder_block(x,         num_filters=128, ker=4) #8
    x, s4 = encoder_block(x,         num_filters=256, ker=3) #4
    x, s5 = encoder_block(x,         num_filters=512, ker=3) #2
    x, s6 = encoder_block(x,         num_filters=1024,ker=3) #1

    # ─── SKIP BLOCKS ────
    s1 = skip_block(s1, ker=3)
    s2 = skip_block(s2, ker=3)
    s3 = skip_block(s3, ker=3)
    s4 = skip_block(s4, ker=3)
    s5 = skip_block(s5, ker=3)
    s6 = skip_block(s6, ker=3)

    # ─── BOTTLENECK ──────────────────────────────────────

    x = Conv3D(2048, 2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)

    # ─── DECODER ─────────────────────────────────────────
    x = decoder_block(x,            num_filters=1024, ker=3, skip_input=None)  # up to 1/32
    x = decoder_block(x,            num_filters=512,  ker=3, skip_input=s6)  # up to 1/16
    x = decoder_block(x,            num_filters=256,  ker=3, skip_input=s5)  # up to 1/8
    x = decoder_block(x,            num_filters=128,  ker=4, skip_input=s4)  # up to 1/4
    x = decoder_block(x,            num_filters=64,   ker=5, skip_input=s3)  # up to 1/2
    x = decoder_block(x,            num_filters=32,   ker=5, skip_input=s2)  # up to full

    # ─── FINAL MERGE ───────────────────────────
    x = Concatenate()([x, s1])
    x = Conv3D(16, 5, strides=1, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)

    output1 = Conv3D(1, 5, padding="same")(x)

    model = Model(inputs, output1, name="PhaseUNet")
    return model

# --- Args ---
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', type=str, default='/data/projects/id01ml/Datasets/Strained/3D_all_strained_64_TRAIN/',
                   help='Directory with training .npz files')
    p.add_argument('--val_dir',   type=str, default='/data/projects/id01ml/Datasets/Strained/3D_all_strained_64_TEST/',
                   help='Directory with validation .npz files')
    p.add_argument('--test_dir',  type=str, default='/data/projects/id01ml/Datasets/Strained/3D_all_strained_64_VALID/',
                   help='Directory with test .npz files')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs',     type=int, default=60)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--dim', nargs=3, type=int, default=[64,64,64])
    p.add_argument('--model_dir',  type=str, default='models')
    p.add_argument('--input_log_data', action='store_true')
    p.add_argument(
       "--initial_weights",
       type=str, default=None,
       help="Path to a checkpoint file to load before training"
    )
    p.add_argument(
       "--initial_epoch",
       type=int, default=0,
       help="Epoch number at which to start training"
    )
    return p.parse_args()

# --- Main ---
def main():
    args = parse_args()
    # gather files
    train_files = sorted(glob.glob(os.path.join(args.train_dir, '*.npz')))
    val_files   = sorted(glob.glob(os.path.join(args.val_dir, '*.npz')))
    test_files  = sorted(glob.glob(os.path.join(args.test_dir, '*.npz'))) if args.test_dir else []

    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(args.model_dir, 'logs', ts)
    ckpt_dir = os.path.join(args.model_dir, 'checkpoints', ts)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_dataset = create_dataset(
        train_files,
        args.batch_size,
        tuple(args.dim),
        input_log_data=args.input_log_data,
        shuffle=True,
        repeat=True  
    )
    
    val_dataset = create_dataset(
        val_files,
        args.batch_size,
        tuple(args.dim),
        input_log_data=args.input_log_data,
        shuffle=False,  
        repeat=False   
    )
    
    strategy = tf.distribute.MirroredStrategy()
    
    with strategy.scope():
        train_dataset = strategy.experimental_distribute_dataset(train_dataset)
        val_dataset = strategy.experimental_distribute_dataset(val_dataset)
        
        model = build_unet(tuple(args.dim)+(1,))
        # model.summary()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.learning_rate),
            loss=loss_combo,
            metrics=[loss_combo]
        )
        if args.initial_weights:
            model.load_weights(args.initial_weights)

    callbacks=[
        TensorBoard(log_dir=log_dir, histogram_freq=1),
        ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, 'ckpt-{epoch:02d}'),
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss'
        )
    ]

    steps_per_epoch = len(train_files) // args.batch_size
    validation_steps = len(val_files) // args.batch_size
    
    num_replicas = strategy.num_replicas_in_sync
    steps_per_epoch = len(train_files) // (args.batch_size * num_replicas)
    validation_steps = len(val_files) // (args.batch_size * num_replicas)
    
    print(f"Number of replicas: {num_replicas}")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Batch size per replica: {args.batch_size}")
    print(f"Global batch size: {args.batch_size * num_replicas}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )
    
if __name__=='__main__':
    main()
