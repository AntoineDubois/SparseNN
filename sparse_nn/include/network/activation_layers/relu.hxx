#pragma once
#include "algebra.hxx"
#include "base_activation_layer.hxx"
#include <iostream>

template<typename T>
class ReLU: public BaseActivationLayer<T>
{
private:
    inline T positive_part(T x);
    inline void inplace_positive_part(T& x);
public:
    ReLU();
    ~ReLU() override;

    Array<T>& operator()(Array<T>& input_array) override;
    Array<T>& Forward(Array<T>& input_array) override;

    inline std::string name() override;
};

#include "relu_impl.hxx"
