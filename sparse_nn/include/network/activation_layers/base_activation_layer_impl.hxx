#include "base_activation_layer.hxx"
#include "algebra.hxx"

#include <iostream>

template<typename T>
BaseActivationLayer<T>::BaseActivationLayer()
{
    // nothing here
}
template<typename T>
BaseActivationLayer<T>::~BaseActivationLayer()
{
    // nothing here
}
template<typename T>
Array<T>& BaseActivationLayer<T>::operator()(Array<T>& input_array)
{
    return input_array;
}
template<typename T>
Array<T>& BaseActivationLayer<T>::Forward(Array<T>& input_array)
{
    return input_array;
}

template<typename T>
inline std::string BaseActivationLayer<T>::name()
{
    return "BaseActivationLayer";
}

template<typename T>
std::ostream& operator<<(std::ostream& out, BaseActivationLayer<T>& layer)
{
    out << "BaseActivationLayer";
    return out;
}