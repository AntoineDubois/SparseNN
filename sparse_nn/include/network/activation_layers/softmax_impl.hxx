#include "softmax.hxx"
#include "algebra.hxx"

#include <cmath>
#include <omp.h>
#include <iostream>


template<typename T>
Softmax<T>::Softmax()
{
    // nothing here
    // computation can be faster if the array were ordered by column instead of row
}
template<typename T>
Softmax<T>::~Softmax()
{
    // nothing here
}

template<typename T>
Array<T>& Softmax<T>::operator()(Array<T>& input_array)
{
    T max_value = input_array.max();
    #pragma omp parallel firstprivate(max_value)
    {
        unsigned column = omp_get_thread_num();
        unsigned nbthreads = omp_get_num_threads();

        unsigned row;
        for(; column < input_array.ncols; column += nbthreads){
            T sum = (T)0;
            for(row=0; row < input_array.nrows; ++row){
                input_array.values_ptr[row * input_array.ncols + column] -= max_value;
                sum += input_array.values_ptr[row * input_array.ncols + column] = std::exp(input_array.values_ptr[row * input_array.ncols + column]);
            }
            sum = 1/sum;
            for(row=0; row < input_array.nrows; ++row){
                input_array.values_ptr[row * input_array.ncols + column] *= sum;
            }
        }
    }
    return input_array;
}
template<typename T>
Array<T>& Softmax<T>::Forward(Array<T>& input_array)
{
    T max_value = input_array.max();
    #pragma omp parallel firstprivate(max_value)
    {
        unsigned column = omp_get_thread_num();
        unsigned nbthreads = omp_get_num_threads();

        unsigned row;
        for(; column < input_array.ncols; column += nbthreads){
            T sum = (T)0;
            for(row=0; row < input_array.nrows; ++row){
                input_array.values_ptr[row * input_array.ncols + column] -= max_value;
                sum += input_array.values_ptr[row * input_array.ncols + column] = std::exp(input_array.values_ptr[row * input_array.ncols + column]);
            }
            sum = 1/sum;
            for(row=0; row < input_array.nrows; ++row){
                input_array.values_ptr[row * input_array.ncols + column] *= sum;
            }
        }
    }
    return input_array;
    
}
template<typename T>
inline std::string Softmax<T>::name()
{
    return "Softmax";
    
}
template<typename T>
std::ostream& operator<<(std::ostream& out, Softmax<T>& layer)
{
    out << "Softmax";
    return out;
}