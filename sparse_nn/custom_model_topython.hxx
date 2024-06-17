
#pragma once
#include "./include/network.hxx"
#include <omp.h>
//#include <pybind11/pybind11.h>
//namespace py = pybind11;

template<typename T>
class CustomNetwork
{

    NeuralNetwork<float> dense_net;
    
Convolution<float>* layer_0;

ReLU<float>* layer_1;

Convolution<float>* layer_2;

ReLU<float>* layer_3;

Max2dPooling<float>* layer_4;

Flatten<float>* layer_5;

Linear<float>* layer_6;

ReLU<float>* layer_7;

Linear<float>* layer_8;

Max<float>* layer_9;

public:
    CustomNetwork(){
        
layer_0= new Convolution<float>(1, 32, 3, 1, 0, 0.0, 0.0) ;
layer_1= new ReLU<float>() ;
layer_2= new Convolution<float>(32, 64, 3, 1, 0, 0.0, 0.0) ;
layer_3= new ReLU<float>() ;
layer_4= new Max2dPooling<float>(2, 2, 0) ;
layer_5= new Flatten<float>() ;
layer_6= new Linear<float>(9216, 128) ;
layer_7= new ReLU<float>() ;
layer_8= new Linear<float>(128, 10) ;
layer_9= new Max<float>() ;
        dense_net.append(layer_0, layer_1);
dense_net.append(layer_2, layer_3);
dense_net.append(layer_4, layer_5);
dense_net.append(layer_6, layer_7);
dense_net.append(layer_8, layer_9);

     };
    Array<float> operator()(Array<float>& input)
    {
        omp_set_num_threads(1);
        return dense_net.operator()(input);
    }
    Array<float> Forward(Array<float>& input)
    {
        omp_set_num_threads(1);
        return dense_net.Forward(input);
    }
    ~CustomNetwork(){
delete layer_0;
delete layer_1;
delete layer_2;
delete layer_3;
delete layer_4;
delete layer_5;
delete layer_6;
delete layer_7;
delete layer_8;
delete layer_9;

    }
};
        