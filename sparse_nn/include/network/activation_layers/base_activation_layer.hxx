#pragma once
#include "algebra.hxx"
#include <iostream>

template<typename T>
class BaseActivationLayer
{
public:
    BaseActivationLayer();
    virtual ~BaseActivationLayer();
    
    virtual Array<T>& operator()(Array<T>& input_array);
    virtual Array<T>& Forward(Array<T>& input_array);

    inline virtual std::string name();
};

#include "base_activation_layer_impl.hxx"