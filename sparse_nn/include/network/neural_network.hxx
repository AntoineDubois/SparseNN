#pragma once
#include <vector>
#include <cassert>
#include "algebra.hxx"
#include "./layers/base_layer.hxx"
#include "./activation_layers/base_activation_layer.hxx"
#include "./activation_layers/identity.hxx"

template<typename T>
class NeuralNetwork
{
public:
    std::vector<BaseLayer<T>*> layers_vec_ptr;
    std::vector<BaseActivationLayer<T>*> activation_layers_vec_ptr;
public:
    unsigned nb_layers;

    NeuralNetwork();
    virtual ~NeuralNetwork();
    void append(BaseLayer<T>* layer, BaseActivationLayer<T>* activation_layer = new Identity<T>());
    void clear();
    
    Array<T> operator()(Array<T> array);
    Array<T> Forward(Array<T> array);

    inline BaseLayer<T>* &layer(unsigned layer) ;
    inline BaseLayer<T>* layer(unsigned layer) const;
    inline BaseActivationLayer<T>* &activation(unsigned layer);
    inline BaseActivationLayer<T>* activation(unsigned layer) const;
};



#include "neural_network_impl.hxx"
