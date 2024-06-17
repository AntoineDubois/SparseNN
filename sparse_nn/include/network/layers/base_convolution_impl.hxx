#include "base_convolution.hxx"
#include <iostream>

template<typename T>
BaseConvolution<T>::BaseConvolution(unsigned n_input_channels, unsigned n_output_channels,
            unsigned kernel_size, unsigned stride, unsigned padding): 
            n_input_channels(n_input_channels), n_output_channels(n_output_channels),
            kernel_size(kernel_size), stride(stride), padding(padding)
{
    checkinputs();
}
template<typename T>
BaseConvolution<T>::~BaseConvolution()
{
    // nothing here
}
template<typename T>
void BaseConvolution<T>::checkinputs()
{
    assert(n_input_channels > 0 && "Number of Input channels must exceed zero");
    assert(n_output_channels > 0 && "Number of Output channels must exceed zero");
    assert(padding < kernel_size && "The padding cannot exceed or be equal to the kernel size");
    assert(padding >= 0 && "The padding cannot be negative");
    assert(stride >= 1 && "The stride must exceed or be equal to 1");
    assert(stride <= kernel_size && "The stride must be smaller than or equal to the kernel size");
}

template<typename T>
inline unsigned BaseConvolution<T>::get_dim(unsigned size)
{
    return ((size + 2 * padding - kernel_size)/ stride +1);
}
template<typename T>
inline unsigned BaseConvolution<T>::get_dim(unsigned size, unsigned bottom_padding, unsigned top_padding)
{
    return ( (size + bottom_padding + top_padding - kernel_size) / stride +1);
}
template<typename T>
void BaseConvolution<T>::copyRowStrip(Array<T>& input_image, Array<T>& input_image_strip, unsigned from_row, unsigned row_strip_size)
{

    unsigned next_row_strip;
    unsigned next_row_input;
    for(unsigned input_channel = 0; input_channel < n_input_channels; ++input_channel){
        for(unsigned row = 0; row < row_strip_size; ++row){
            next_row_strip = input_channel * row_strip_size + row * input_image.ncols;
            next_row_input = input_channel * input_image.nrows + row * input_image.ncols + from_row;
            for(unsigned column = 0; column < n_input_cols; ++column){
                input_image_strip.values_ptr[next_row_strip + column] = input_image.values_ptr[next_row_input + column];
            }
        }
    }
}
template<typename T>
inline std::string BaseConvolution<T>::name()
{
    return "BaseConvolution";
}
template<typename T>
std::ostream& operator<<(std::ostream& out, BaseConvolution<T>& layer)
{
    out << "(k=" << layer.kernel_size << ", s=" << layer.stride << ", p=" << layer.padding << ", in_ch=" << layer.n_input_channels << ", out_ch=" << layer.n_output_channels;
    return out;
}