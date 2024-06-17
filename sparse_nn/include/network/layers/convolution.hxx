#pragma once
#include "base_convolution.hxx"
#include "algebra.hxx"

#include <vector>
#include <iostream>

template<typename T>
class Convolution: public BaseConvolution<T>
{
public:
    std::vector<Array<T>> kernels;
    std::vector<T> biases;

    Convolution(unsigned n_input_channels, unsigned n_output_channels,
                unsigned kernel_size, unsigned stride, unsigned padding,
                T kernel_init_value, T bias_init_value);
    template<typename Generator>
    Convolution(unsigned n_input_channels, unsigned n_output_channels,
                unsigned kernel_size, unsigned stride, unsigned padding,
                Generator& lambda_kernels, T bias_init_value);
    template<typename Generator>
    Convolution(unsigned n_input_channels, unsigned n_output_channels,
                unsigned kernel_size, unsigned stride, unsigned padding,
                T kernel_init_value, Generator& lambda_bias);
    template<typename Generator>
    Convolution(unsigned n_input_channels, unsigned n_output_channels,
                unsigned kernel_size, unsigned stride, unsigned padding,
                Generator& lambda_kernels, Generator& lambda_bias);
    
    ~Convolution() override;
    
    Array<T> operator()(Array<T>& input_image) override;
    Array<T> Forward(Array<T>& input_image) override;

    inline std::string name() override;
private:
    T convolution_product(Array<T>& input, Array<T>& out_channel_kernel, unsigned row, unsigned column, unsigned bottom_padding, unsigned top_padding);
};

#include "convolution_impl.hxx"