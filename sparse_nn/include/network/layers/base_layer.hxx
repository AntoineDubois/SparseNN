#pragma once
#include "algebra.hxx"
#include <iostream>

template<typename T>
class BaseLayer
{
public:
    BaseLayer();
    virtual ~BaseLayer();

    virtual Array<T> operator()(Array<T>& input_array);
    virtual Array<T> Forward(Array<T>& input_array);
    
    inline virtual std::string name();
};

#include "base_layer_impl.hxx"