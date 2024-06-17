#pragma once
#include "max_2d_pooling.hxx"
#include "algebra.hxx"

#include <cassert>
#include <algorithm>
#include <omp.h>
#include <iostream>
#include <limits>

template<typename T>
Max2dPooling<T>::Max2dPooling(unsigned kernel_size, unsigned stride, unsigned padding): 
            BaseConvolution<T>(1, 1, kernel_size, stride, padding)
{
    // nothing here
}
template<typename T>
Max2dPooling<T>::~Max2dPooling()
{
    // nothing here
}
template<typename T>
Array<T> Max2dPooling<T>::operator()(Array<T>& input_image)
{
    BaseConvolution<T>::n_input_channels = input_image.nchannels;
    BaseConvolution<T>::n_input_rows = input_image.nrows;
    BaseConvolution<T>::n_input_cols = input_image.ncols;
    BaseConvolution<T>::n_output_channels = BaseConvolution<T>::n_input_channels;
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
                for(unsigned row = 0; row < n_output_rows_strip; ++row){
                    for(unsigned column = 0; column < BaseConvolution<T>::n_output_cols; ++column){
                        output_image(output_channel, row + from_output_row, column) = maxBlock(input_image_strip, output_channel, row, column, bottom_padding, top_padding);
                    }
                }
            }
        }
    }
    return output_image;
}
template<typename T>
Array<T> Max2dPooling<T>::Forward(Array<T>& input_image)
{
    BaseConvolution<T>::n_input_channels = input_image.nchannels;
    BaseConvolution<T>::n_input_rows = input_image.nrows;
    BaseConvolution<T>::n_input_cols = input_image.ncols;
    BaseConvolution<T>::n_output_channels = BaseConvolution<T>::n_input_channels;
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
                for(unsigned row = 0; row < n_output_rows_strip; ++row){
                    for(unsigned column = 0; column < BaseConvolution<T>::n_output_cols; ++column){
                        output_image(output_channel, row + from_output_row, column) = maxBlock(input_image_strip, output_channel, row, column, bottom_padding, top_padding);
                    }
                }
            }
        }
    }
    return output_image;
}

template<typename T>
T Max2dPooling<T>::maxBlock(Array<T>& input, unsigned channel, unsigned row, unsigned column, unsigned bottom_padding, unsigned top_padding)
{   
    int row_prime_low = row * BaseConvolution<T>::stride -top_padding; 
    unsigned row_start = std::max(int(top_padding -row), int(0));
    unsigned row_end = BaseConvolution<T>::kernel_size;
    if ( input.nrows < row_prime_low + BaseConvolution<T>::kernel_size ) row_end = input.nrows -row_prime_low;
    unsigned u;
    

    int column_prime = column * BaseConvolution<T>::stride -BaseConvolution<T>::padding;
    unsigned column_start = std::max(int(BaseConvolution<T>::padding -column), int(0));
    unsigned column_end = BaseConvolution<T>::kernel_size;
    if(input.ncols < column_prime + BaseConvolution<T>::kernel_size ) column_end = input.ncols -column_prime;
    unsigned v;
    
    T max_element = std::numeric_limits<T>::lowest();
    double element;
    unsigned next_row;
    for(u = row_prime_low +row_start; u < row_prime_low + row_end; ++u){
        next_row = (channel + u) * BaseConvolution<T>::kernel_size;
        for(v = column_prime + column_start; v < column_prime + column_end; ++v){
            element = input.values_ptr[next_row + v];
            if (element > max_element){
                max_element = element;
            }
        }
    }
    
    return max_element;
}
template<typename T>
inline std::string Max2dPooling<T>::name()
{
    return "Max2dPooling";
}
template<typename T>
std::ostream& operator<<(std::ostream& out, Max2dPooling<T>& layer)
{
    out << "Max2dPooling" << static_cast<BaseConvolution<T>&>(layer);
    return out;
}