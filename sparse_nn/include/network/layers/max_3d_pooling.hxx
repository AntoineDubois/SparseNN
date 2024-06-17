#pragma once
#include "base_convolution.hxx"
#include "algebra.hxx"
#include <iostream>


template<typename T>
class Max3dPooling: public BaseConvolution<T>
{
public:
    Max3dPooling(unsigned kernel_size, unsigned stride, unsigned padding);
    ~Max3dPooling() override;
    
    Array<T> operator()(Array<T>& input_image) override;
    Array<T> Forward(Array<T>& input_image) override;

    inline std::string name() override;
private:
    T maxBlock(Array<T>& input, unsigned row, unsigned column, unsigned bottom_padding, unsigned top_padding);
};

#include "max_3d_pooling_impl.hxx"