#include "sparse_convolution.hxx"
#include "convolution.hxx"
#include "algebra.hxx"

#include <cassert>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <iostream>

template<typename T>
SparseConvolution<T>::SparseConvolution(Convolution<T>& convolution_layer): 
            BaseConvolution<T>(convolution_layer.n_input_channels, convolution_layer.n_output_channels, convolution_layer.kernel_size, convolution_layer.stride, convolution_layer.padding)
{
    std::vector<SparseArray<T>> kernels_loc;
    for(unsigned channel = 0; channel < BaseConvolution<T>::n_output_channels; ++channel)
    {
        kernels.push_back(SparseArray<T>(convolution_layer.kernels[channel]));
        biases.push_back(convolution_layer.biases[channel]);
    }
}
template<typename T>
SparseConvolution<T>::~SparseConvolution()
{
    // nothing here
}

template<typename T>
Array<T> SparseConvolution<T>::operator()(Array<T>& input_image)
{
    assert(input_image.nchannels == BaseConvolution<T>::n_input_channels);

    BaseConvolution<T>::n_input_rows = input_image.nrows;
    BaseConvolution<T>::n_input_cols = input_image.ncols;
    assert(BaseConvolution<T>::n_input_rows >= BaseConvolution<T>::kernel_size  && "The input image must be larger than the kernel width");
    assert(BaseConvolution<T>::n_input_cols >= BaseConvolution<T>::kernel_size && "The input image must be taller than the kernel height");
    BaseConvolution<T>::n_output_rows = BaseConvolution<T>::get_dim(BaseConvolution<T>::n_input_rows);
    BaseConvolution<T>::n_output_cols = BaseConvolution<T>::get_dim(BaseConvolution<T>::n_input_cols);

    Array<T> output_image(BaseConvolution<T>::n_output_channels, BaseConvolution<T>::n_output_rows, BaseConvolution<T>::n_output_cols);
    
    unsigned number_strips = BaseConvolution<T>::n_input_rows / BaseConvolution<T>::kernel_size;
    // first strip
    unsigned n_output_rows_first_strip = (BaseConvolution<T>::kernel_size + BaseConvolution<T>::padding -1)/ BaseConvolution<T>::stride +1;
    unsigned init_next_after_first = n_output_rows_first_strip * BaseConvolution<T>::stride;
    unsigned input_span_first_strip = (n_output_rows_first_strip -1) * BaseConvolution<T>::stride + BaseConvolution<T>::kernel_size;

    // middle strips
    unsigned n_output_rows_middle_strip = (BaseConvolution<T>::kernel_size -1)/BaseConvolution<T>::stride +1;
    unsigned init_next_after_middle = n_output_rows_middle_strip * BaseConvolution<T>::stride;
    unsigned input_span_middle_strip = (n_output_rows_middle_strip -1) * BaseConvolution<T>::stride + BaseConvolution<T>::kernel_size;
    // last strip
    unsigned init_row_last_strip = 0 + (number_strips -1) * init_next_after_middle;
    unsigned input_span_last_strip = BaseConvolution<T>::n_input_rows -init_row_last_strip;
    unsigned n_output_rows_last_strip = (input_span_last_strip + BaseConvolution<T>::padding -BaseConvolution<T>::kernel_size) / BaseConvolution<T>::stride +1;


    #pragma omp parallel
    {
        unsigned row_thread_nb = omp_get_thread_num();
        unsigned num_threads = omp_get_num_threads();

        unsigned init_input_row_strip;
        unsigned input_span;
        unsigned n_output_rows_strip;
        unsigned top_padding;
        unsigned bottom_padding;
        
        unsigned from_output_row;
        
        T loc_bias;
        for(; row_thread_nb < number_strips; row_thread_nb += num_threads )
        {
            init_input_row_strip = init_next_after_first + (row_thread_nb -1) * init_next_after_middle;
            input_span = input_span_middle_strip;
            n_output_rows_strip = n_output_rows_middle_strip;
            from_output_row = n_output_rows_first_strip + (row_thread_nb -1) * n_output_rows_middle_strip;
            top_padding = 0;
            bottom_padding = 0;
            if (row_thread_nb == 0){ // Is first strip ?
                init_input_row_strip = 0;
                input_span = input_span_first_strip;
                n_output_rows_strip = n_output_rows_first_strip;
                from_output_row = 0;
                top_padding = BaseConvolution<T>::padding;
                if (row_thread_nb == number_strips -1){  // Is also last strip ?
                    n_output_rows_strip = BaseConvolution<T>::n_output_rows;
                    input_span = BaseConvolution<T>::n_input_rows;
                    bottom_padding = BaseConvolution<T>::padding;
                }
            } else if (row_thread_nb == number_strips -1){ // Is last strip ?
                init_input_row_strip = init_row_last_strip;
                input_span = input_span_last_strip;
                n_output_rows_strip = n_output_rows_last_strip;
                bottom_padding = BaseConvolution<T>::padding;
            }

            if ( n_output_rows_strip == 0) continue;

            Array<T> input_image_strip(BaseConvolution<T>::n_input_channels, input_span, BaseConvolution<T>::n_input_cols);
            BaseConvolution<T>::copyRowStrip(input_image, input_image_strip, init_input_row_strip, input_span);

            for(unsigned output_channel = 0; output_channel < BaseConvolution<T>::n_output_channels; ++output_channel)
            {
                SparseArray<T> out_channel_kernel(kernels[output_channel]);
                loc_bias = biases[output_channel];
                for(unsigned row = 0; row < n_output_rows_strip; ++row){
                    for(unsigned column = 0; column < BaseConvolution<T>::n_output_cols; ++column){
                        output_image(output_channel, row + from_output_row, column) = convolution_product(input_image_strip, out_channel_kernel, row, column, bottom_padding, top_padding) + loc_bias;
                    }
                }
            }
        }
    }
    return output_image;
}
template<typename T>
Array<T> SparseConvolution<T>::Forward(Array<T>& input_image)
{
    assert(input_image.nchannels == BaseConvolution<T>::n_input_channels);

    BaseConvolution<T>::n_input_rows = input_image.nrows;
    BaseConvolution<T>::n_input_cols = input_image.ncols;
    assert(BaseConvolution<T>::n_input_rows >= BaseConvolution<T>::kernel_size  && "The input image must be larger than the kernel width");
    assert(BaseConvolution<T>::n_input_cols >= BaseConvolution<T>::kernel_size && "The input image must be taller than the kernel height");
    BaseConvolution<T>::n_output_rows = BaseConvolution<T>::get_dim(BaseConvolution<T>::n_input_rows);
    BaseConvolution<T>::n_output_cols = BaseConvolution<T>::get_dim(BaseConvolution<T>::n_input_cols);

    Array<T> output_image(BaseConvolution<T>::n_output_channels, BaseConvolution<T>::n_output_rows, BaseConvolution<T>::n_output_cols);
    
    unsigned number_strips = BaseConvolution<T>::n_input_rows / BaseConvolution<T>::kernel_size;
    // first strip
    unsigned n_output_rows_first_strip = (BaseConvolution<T>::kernel_size + BaseConvolution<T>::padding -1)/ BaseConvolution<T>::stride +1;
    unsigned init_next_after_first = n_output_rows_first_strip * BaseConvolution<T>::stride;
    unsigned input_span_first_strip = (n_output_rows_first_strip -1) * BaseConvolution<T>::stride + BaseConvolution<T>::kernel_size;

    // middle strips
    unsigned n_output_rows_middle_strip = (BaseConvolution<T>::kernel_size -1)/BaseConvolution<T>::stride +1;
    unsigned init_next_after_middle = n_output_rows_middle_strip * BaseConvolution<T>::stride;
    unsigned input_span_middle_strip = (n_output_rows_middle_strip -1) * BaseConvolution<T>::stride + BaseConvolution<T>::kernel_size;
    // last strip
    unsigned init_row_last_strip = 0 + (number_strips -1) * init_next_after_middle;
    unsigned input_span_last_strip = BaseConvolution<T>::n_input_rows -init_row_last_strip;
    unsigned n_output_rows_last_strip = (input_span_last_strip + BaseConvolution<T>::padding -BaseConvolution<T>::kernel_size) / BaseConvolution<T>::stride +1;


    #pragma omp parallel
    {
        unsigned row_thread_nb = omp_get_thread_num();
        unsigned num_threads = omp_get_num_threads();

        unsigned init_input_row_strip;
        unsigned input_span;
        unsigned n_output_rows_strip;
        unsigned top_padding;
        unsigned bottom_padding;
        
        unsigned from_output_row;

        T loc_bias;
        for(; row_thread_nb < number_strips; row_thread_nb += num_threads )
        {
            init_input_row_strip = init_next_after_first + (row_thread_nb -1) * init_next_after_middle;
            input_span = input_span_middle_strip;
            n_output_rows_strip = n_output_rows_middle_strip;
            from_output_row = n_output_rows_first_strip + (row_thread_nb -1) * n_output_rows_middle_strip;
            top_padding = 0;
            bottom_padding = 0;
            if (row_thread_nb == 0){ // Is first strip ?
                init_input_row_strip = 0;
                input_span = input_span_first_strip;
                n_output_rows_strip = n_output_rows_first_strip;
                from_output_row = 0;
                top_padding = BaseConvolution<T>::padding;
                if (row_thread_nb == number_strips -1){  // Is also last strip ?
                    n_output_rows_strip = BaseConvolution<T>::n_output_rows;
                    input_span = BaseConvolution<T>::n_input_rows;
                    bottom_padding = BaseConvolution<T>::padding;
                }
            } else if (row_thread_nb == number_strips -1){ // Is last strip ?
                init_input_row_strip = init_row_last_strip;
                input_span = input_span_last_strip;
                n_output_rows_strip = n_output_rows_last_strip;
                bottom_padding = BaseConvolution<T>::padding;
            }

            if ( n_output_rows_strip == 0) continue;

            Array<T> input_image_strip(BaseConvolution<T>::n_input_channels, input_span, BaseConvolution<T>::n_input_cols);
            BaseConvolution<T>::copyRowStrip(input_image, input_image_strip, init_input_row_strip, input_span);

            for(unsigned output_channel = 0; output_channel < BaseConvolution<T>::n_output_channels; ++output_channel)
            {
                SparseArray<T> out_channel_kernel(kernels[output_channel]);
                loc_bias = biases[output_channel];
                for(unsigned row = 0; row < n_output_rows_strip; ++row){
                    for(unsigned column = 0; column < BaseConvolution<T>::n_output_cols; ++column){
                        output_image(output_channel, row + from_output_row, column) = convolution_product(input_image_strip, out_channel_kernel, row, column, bottom_padding, top_padding) + loc_bias;
                    }
                }
            }
        }
    }
    return output_image;
}

template<typename T>
T SparseConvolution<T>::convolution_product(Array<T>& input, SparseArray<T>& out_channel_kernel, unsigned row, unsigned column, unsigned bottom_padding, unsigned top_padding)
{   
    int row_prime_low = row * BaseConvolution<T>::stride -top_padding; 
    unsigned row_start = std::max(int(top_padding -row), int(0));
    unsigned row_end = BaseConvolution<T>::kernel_size;
    if ( input.nrows < row_prime_low + BaseConvolution<T>::kernel_size ) row_end = input.nrows -row_prime_low;
    unsigned u;
    unsigned u_prime;

    int column_prime = column * BaseConvolution<T>::stride -BaseConvolution<T>::padding;
    unsigned column_start = std::max(int(BaseConvolution<T>::padding -column), int(0));
    unsigned column_end = BaseConvolution<T>::kernel_size;
    if(input.ncols < column_prime + BaseConvolution<T>::kernel_size ) column_end = input.ncols -column_prime;
    
    unsigned k;
    T sum = (T)0;
    
    unsigned col_loc;
    unsigned next_row_kernel = 0;
    unsigned next_row_input;
    for(unsigned input_channel = 0; input_channel < BaseConvolution<T>::n_input_channels; ++input_channel){
        for(u = row_start; u < row_end; ++u){
            next_row_input = input_channel * input.nrows + (row_prime_low +u) * input.ncols + column_prime;
            for(k = out_channel_kernel.row_cum_ptr[next_row_kernel + u]; k < out_channel_kernel.row_cum_ptr[next_row_kernel + u +1]; ++k){
                col_loc = out_channel_kernel.col_index_ptr[k];
                if( (col_loc < column_start) || (col_loc >= column_end) ) continue;
                sum += out_channel_kernel.values_ptr[k] * input.values_ptr[next_row_input + col_loc];
            }
        }
        next_row_kernel += BaseConvolution<T>::kernel_size;
    }
    
    return sum;
}
template<typename T>
inline std::string SparseConvolution<T>::name()
{
    return "SparseConvolution";
}
template<typename T>
std::ostream& operator<<(std::ostream& out, SparseConvolution<T>& layer)
{
    out << "SparseConvolution" << static_cast<BaseConvolution<T>&>(layer);
    return out;
}
