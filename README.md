# Sparse Neural Network
This repository is a sketch of a sparse neural network library (inference only). The danse and sparse matrix multiplication, and all the dense and sparse layers are parallelised. I wanted to develop my skills in parallelisation.
**Take what you want from it!**

Building the project:
*for python binding*: 
- cd sparse_nn
- git clone https://github.com/pybind/pybind11.git
- mkdir build
- cd build
- cmake ..
- make

*for c++ executable*: 
- cd sparse_nn
- mkdir build
- cd build
- cmake .. -DBUILD_PYBIND=FALSE
- make