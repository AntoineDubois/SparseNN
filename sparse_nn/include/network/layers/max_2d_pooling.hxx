#pragma once
#include "base_convolution.hxx"
#include "algebra.hxx"
#include <iostream>

template<typename T>
class Max2dPooling: public BaseConvolution<T>
{
public:
    Max2dPooling(unsigned kernel_size, unsigned stride, unsigned padding);
    ~Max2dPooling() override;
    
    Array<T> operator()(Array<T>& input_image) override;
    Array<T> Forward(Array<T>& input_image) override;
    
    inline std::string name() override;
private:
    T maxBlock(Array<T>& input, unsigned channel, unsigned row, unsigned column, unsigned bottom_padding, unsigned top_padding);
};

#include "max_2d_pooling_impl.hxx"