#pragma once
#include "base_activation_layer.hxx"
#include "algebra.hxx"
#include <iostream>

template<typename T>
class Flatten: public BaseActivationLayer<T>
{
private:
    // nothing here
public:
    Flatten();
    ~Flatten() override;
    
    inline Array<T>& operator()(Array<T>& array) override;
    Array<T>& Forward(Array<T>& array) override;

    inline std::string name() override;
};

#include "flatten_impl.hxx"