#include "logsoftmax.hxx"
#include "algebra.hxx"

#include <cmath>
#include <omp.h>
#include <iostream>


template<typename T>
LogSoftmax<T>::LogSoftmax()
{
    // nothing here
}
template<typename T>
LogSoftmax<T>::~LogSoftmax()
{
    // nothing here
}

template<typename T>
Array<T>& LogSoftmax<T>::operator()(Array<T>& input_array)
{
    #pragma omp parallel
    {
        unsigned column = omp_get_thread_num();
        unsigned nbthreads = omp_get_num_threads();

        unsigned row;
        for(; column < input_array.ncols; column += nbthreads){
            T sum = (T)0;
            for(row=0; row < input_array.nrows; ++row){
                sum += std::exp(input_array.values_ptr[row * input_array.ncols + column]);
            }
            sum = -std::log(sum);
            for(row=0; row < input_array.nrows; ++row){
                input_array.values_ptr[row * input_array.ncols + column] += sum;
            }
        }
    }
    return input_array;
}
template<typename T>
Array<T>& LogSoftmax<T>::Forward(Array<T>& input_array)
{
    #pragma omp parallel
    {
        unsigned column = omp_get_thread_num();
        unsigned nbthreads = omp_get_num_threads();

        unsigned row;
        for(; column < input_array.ncols; column += nbthreads){
            T sum = (T)0;
            for(row=0; row < input_array.nrows; ++row){
                sum += std::exp(input_array.values_ptr[row * input_array.ncols + column]);
            }
            sum = -std::log(sum);
            for(row=0; row < input_array.nrows; ++row){
                input_array.values_ptr[row * input_array.ncols + column] += sum;
            }
        }
    }
    return input_array;
}
template<typename T>
inline std::string LogSoftmax<T>::name()
{
    return "LogSoftmax";
}

template<typename T>
std::ostream& operator<<(std::ostream& out, LogSoftmax<T>& layer)
{
    out << "LogSoftmax";
    return out;
}