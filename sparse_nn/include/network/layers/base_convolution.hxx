#pragma once
#include "base_layer.hxx"
#include "algebra.hxx"
#include <iostream>


template<typename T>
class BaseConvolution: public BaseLayer<T>
{
protected:
    unsigned n_output_rows; 
    unsigned n_output_cols;
public:
    unsigned n_input_channels;
    unsigned n_input_rows;
    unsigned n_input_cols;
    unsigned n_output_channels;
    unsigned kernel_size;
    unsigned stride;
    unsigned padding;

    BaseConvolution(unsigned n_input_channels, unsigned n_output_channels,
                unsigned kernel_size, unsigned stride, unsigned padding);
    virtual ~BaseConvolution() override;
    inline virtual std::string name() override;
private:
    void checkinputs();
protected:
    inline unsigned get_dim(unsigned size);
    inline unsigned get_dim(unsigned size, unsigned bottom_padding, unsigned top_padding);
    void copyRowStrip(Array<T>& input_image, Array<T>& input_image_thread, unsigned from_row, unsigned row_strip_size);
};

#include "base_convolution_impl.hxx"