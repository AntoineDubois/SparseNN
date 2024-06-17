#include "relu.hxx"
#include "algebra.hxx"
#include <omp.h>
#include <iostream>


template<typename T>
ReLU<T>::ReLU()
{
    // nothing here
}
template<typename T>
ReLU<T>::~ReLU()
{
    // nothing here
}

template<typename T>
Array<T>& ReLU<T>::operator()(Array<T>& input_array)
{
    #pragma omp parallel
    {
        unsigned element = omp_get_thread_num();
        unsigned nbthreads = omp_get_num_threads();

        for(;element < input_array.size; element += nbthreads)
        {
            inplace_positive_part(input_array.values_ptr[element]);
        } 
    }
    return input_array;
}
template<typename T>
Array<T>& ReLU<T>::Forward(Array<T>& input_array)
{
    #pragma omp parallel
    {
        unsigned element = omp_get_thread_num();
        unsigned nbthreads = omp_get_num_threads();

        for(;element < input_array.size; element += nbthreads)
        {
            inplace_positive_part(input_array.values_ptr[element]);
        } 
    }
    return input_array;
}

template<typename T>
inline T ReLU<T>::positive_part(T x)
{
    return x > (T)0 ? x : T(0);
}
template<typename T>
inline void ReLU<T>::inplace_positive_part(T& x)
{
    if( x < (T)0 ) x = 0;
}
template<typename T>
inline std::string ReLU<T>::name()
{
    return "ReLU";
}
template<typename T>
std::ostream& operator<<(std::ostream& out, ReLU<T>& layer)
{
    out << "ReLU";
    return out;
}
