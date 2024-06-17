#pragma once
#include "base_layer.hxx"
#include "algebra.hxx"
#include <iostream>

template<typename T>
class Linear: public BaseLayer<T>
{
public:
    Array<T> weights;
    Array<T> bias;
    Linear(unsigned input_size, unsigned output_size);
    Linear(unsigned input_size, unsigned output_size, T init_value_weights, T init_value_bias);
    template<typename Generator>
    Linear(unsigned input_size, unsigned output_size, Generator && lambda_weights, T init_value_bias);
    template<typename Generator>
    Linear(unsigned input_size, unsigned output_size, T init_value_weights, Generator && lambda_bias);
    template<typename Generator>
    Linear(unsigned input_size, unsigned output_size, Generator && lambda_weights, Generator && lambda_bias);
    
    ~Linear() override;

    Array<T> operator()(Array<T>& input) override;
    Array<T> Forward(Array<T>& input) override;

    inline std::string name() override;
};

#include "linear_impl.hxx"