#include "neural_network.hxx"
#include "./layers/base_layer.hxx"
#include "./activation_layers/base_activation_layer.hxx"
#include "./activation_layers/identity.hxx"

#include <iostream>
#include <omp.h>

template<typename T>
NeuralNetwork<T>::NeuralNetwork(/* args */): nb_layers(0)
{
    // nothing here
}
template<typename T>
NeuralNetwork<T>::~NeuralNetwork()
{
    // nothing here
}

template<typename T>
void NeuralNetwork<T>::append(BaseLayer<T>* layer_ptr, BaseActivationLayer<T>* activation_layer_ptr)
{
    layers_vec_ptr.push_back(layer_ptr);
    activation_layers_vec_ptr.push_back(activation_layer_ptr);
    ++nb_layers;
}
template<typename T>
void NeuralNetwork<T>::clear()
{
    layers_vec_ptr.clear();
    activation_layers_vec_ptr.clear();
    nb_layers = 0;
}
template<typename T>
Array<T> NeuralNetwork<T>::operator()(Array<T> array)
{
    assert(nb_layers > 0 && "The network must have at least one layer");

    for(unsigned layer = 0; layer < nb_layers; ++layer)
    {    
        array = layers_vec_ptr[layer] -> operator()(array);
        array = std::move( activation_layers_vec_ptr[layer] -> operator()(array) );
    }
    return array;
}
template<typename T>
Array<T> NeuralNetwork<T>::Forward(Array<T> array)
{
    assert(nb_layers > 0 && "The network must have at least one layer");
    
    for(unsigned layer = 0; layer < nb_layers; ++layer)
    {
        array = layers_vec_ptr[layer] -> Forward(array);
        array = std::move( activation_layers_vec_ptr[layer] -> Forward(array) );
    }
    return array;
}
template<typename T>
inline BaseLayer<T>* & NeuralNetwork<T>::layer(unsigned layer)
{
    assert(layer < nb_layers);
    return layers_vec_ptr[layer];
}
template<typename T>
inline BaseLayer<T>* NeuralNetwork<T>::layer(unsigned layer) const
{
    assert(layer < nb_layers);
    return layers_vec_ptr[layer];
}
template<typename T>
inline BaseActivationLayer<T>* & NeuralNetwork<T>::activation(unsigned layer)
{
    assert(layer < nb_layers);
    return activation_layers_vec_ptr[layer];
}
template<typename T>
inline BaseActivationLayer<T>* NeuralNetwork<T>::activation(unsigned layer) const
{
    assert(layer < nb_layers);
    return activation_layers_vec_ptr[layer];
}

template<typename T>
std::ostream& operator<<(std::ostream& out, NeuralNetwork<T>& network)
{
    for(unsigned layer = 0; layer < network.nb_layers; ++layer)
    {
        out << layer << "\t" << network.layer(layer) -> name();
        out << "\n\t" << network.activation(layer) -> name() << "\n";
    }
    return out;
}