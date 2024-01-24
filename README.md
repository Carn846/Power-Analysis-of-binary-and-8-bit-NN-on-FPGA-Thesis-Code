# FPGA-Thesis-Code
Code for my thesis for Power Analysis of 8 bit and Binary Neural Networks for Image Classification on FPGAs

This folder contains the different aspects of the code. It is broken up into the following sub folders:
- FPGA Networks: A folder containing the zip files needed to be deployed to the FPGA (PYNQ-Z2)
- TestSynth: A folder that will contain the Generated Netron Models
- CIFAR-10 Synthesis: the jupter note book that needs to be run inside the docker container. Will Synthesis the provided trinary and 8-bit network that you provide it. Care must be take to ensure that the correct network information and transformation steps are set up (they will be slightly different for different networks).
- MNIST Synthesis: The same as the above but for MNIST
- findBestFolding: A script to determine the optimal target_cycles_per_frame. Only works for the MNIST data set
 
