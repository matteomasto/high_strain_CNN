# high_strain_CNN
A supervised convolutional neural network for the phase retrieval of highly strained Bragg Coherent Diffraction Patterns. 
![image](https://github.com/user-attachments/assets/cdddaf11-2502-485b-b618-db3a65dc48a7)

## About
This model provides an estimate of the reciprocal space phase corresponding to the input BCDI pattern. From the retrieved phase one can obtain the reconstructed object in real space with an inverse Fourier Transform. 
The model is supposed to help finding an initial estimate of the object's shape and phase. A refinement using conventional iterative phase retrieval algorithms can be performed for higher quality. 
The model is written using Tensorflow library v2.10.1

## Instructions
The model accepts as inputs BCDI patterns centered around the center of mass, resized to a 64x64x64 pixels grid, transformed in logarithmic scale and normalized between 0 and 1. 
Download the `model_paper.h5` for direct use of the pretraied model or `train.py` file for the training on your dataset. 

## Funding
I developped the codes during my PhD at the University Grenoble - Alpes and at the ID01 beamline of the European Synchrotron Radiation Facility (ESRF-EBS). The PhD is also part of the ENGAGE programme, thus partially funded by the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement number 101034267.
