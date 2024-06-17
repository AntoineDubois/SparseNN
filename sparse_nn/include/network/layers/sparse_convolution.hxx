#pragma once
#include "algebra.hxx"
#include "convolution.hxx"

#include <vector>
#include <iostream>

template<typename T>
class SparseConvolution: public BaseConvolution<T>
{
private:
    unsigned n_output_rows; 
    unsigned n_output_cols;
public:
    std::vector<SparseArray<T>> kernels;
    std::vector<T> biases;

    SparseConvolution(Convolution<T>& convolution_layer);
    ~SparseConvolution() override;
    
    Array<T> operator()(Array<T>& input_image) override;
    Array<T> Forward(Array<T>& input_image) override;

    inline std::string name() override;
private:
    T convolution_product(Array<T>& input, SparseArray<T>& out_channel_kernel, unsigned row, unsigned column, unsigned bottom_padding, unsigned top_padding);
};

#include "sparse_convolution_impl.hxx"