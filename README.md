This repository is a sketch of a sparse neural network library.
Take what you want from it!

Building the project:
*for python binding*: 
cd sparse_nn
mkdir build
cd build
cmake ..
make

*for c++ executable*: 
cd sparse_nn
mkdir build
cd build
cmake .. -DBUILD_PYBIND=FALSE
make