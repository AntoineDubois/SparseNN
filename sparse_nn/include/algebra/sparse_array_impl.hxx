#include "sparse_array.hxx"
#include "array.hxx"
#include "utils.hxx"

#include <iostream>
#include <cassert>
#include <cmath>
#include <omp.h>


template<typename T> 
SparseArray<T>::SparseArray(Array<T>& array, T tol): BaseArray<T>(array.nchannels, array.nrows, array.ncols, array.size), nb_not_zeros(0)
{
    for(unsigned element = 0; element < array.size; ++element)
    {
        if(std::abs(array(element)) > tol) ++nb_not_zeros;
    }
    values_ptr = new T[nb_not_zeros];
    col_index_ptr = new unsigned[nb_not_zeros];
    row_cum_ptr = new unsigned[BaseArray<T>::nchannels * BaseArray<T>::nrows +1];
    
    fill(array, tol);
}

template<typename T> 
SparseArray<T>::SparseArray(SparseArray<T>& other): BaseArray<T>(other.nchannels, other.nrows, other.ncols, other.size), nb_not_zeros(other.nb_not_zeros)
{
    values_ptr = new T[nb_not_zeros];
    col_index_ptr = new unsigned[nb_not_zeros];
    row_cum_ptr = new unsigned[BaseArray<T>::nchannels * BaseArray<T>::nrows +1];

    fill(other);
}
template<typename T> 
SparseArray<T>::SparseArray(const SparseArray<T>& other): BaseArray<T>(other.nchannels, other.nrows, other.ncols, other.size), nb_not_zeros(other.nb_not_zeros)
{
    values_ptr = new T[nb_not_zeros];
    col_index_ptr = new unsigned[nb_not_zeros];
    row_cum_ptr = new unsigned[BaseArray<T>::nchannels * BaseArray<T>::nrows +1];

    fill(other);
}
template<typename T>
SparseArray<T>::SparseArray(SparseArray<T>&& other): BaseArray<T>(std::move(other.nchannels), std::move(other.nrows), std::move(other.ncols), std::move(other.size)), nb_not_zeros(std::move(other.nb_not_zeros))
{
    values_ptr = std::exchange(other.values_ptr, nullptr);
    col_index_ptr = std::exchange(other.col_index_ptr, nullptr);
    row_cum_ptr = std::exchange(other.row_cum_ptr, nullptr);
}
template<typename T>
SparseArray<T>::SparseArray(const SparseArray<T>&& other): BaseArray<T>(std::move(other.nchannels), std::move(other.nrows), std::move(other.ncols), std::move(other.size)), nb_not_zeros(std::move(other.nb_not_zeros))
{
    values_ptr = std::exchange(other.values_ptr, nullptr);
    col_index_ptr = std::exchange(other.col_index_ptr, nullptr);
    row_cum_ptr = std::exchange(other.row_cum_ptr, nullptr);
}

template<typename T>
SparseArray<T>& SparseArray<T>::operator=(SparseArray& other) noexcept
{
    if ( this == &other ) return *this;

    delete[] values_ptr;
    delete[] col_index_ptr;
    delete[] row_cum_ptr;
    BaseArray<T>::nchannels = other.nchannels;
    BaseArray<T>::nrows = other.nrows;
    BaseArray<T>::ncols = other.ncols;
    BaseArray<T>::size = other.size;
    nb_not_zeros = other.nb_not_zeros;

    values_ptr = new T[nb_not_zeros];
    col_index_ptr = new unsigned[nb_not_zeros];
    row_cum_ptr = new unsigned[BaseArray<T>::nchannels * BaseArray<T>::nrows +1];
    fill(other);
    
    return *this;
}
template<typename T>
SparseArray<T>& SparseArray<T>::operator=(SparseArray&& other) noexcept
{
    if ( this == &other ) return *this;

    delete[] values_ptr;
    delete[] col_index_ptr;
    delete[] row_cum_ptr;
    BaseArray<T>::nchannels = std::move(other.nchannels);
    BaseArray<T>::nrows = std::move(other.nrows);
    BaseArray<T>::ncols = std::move(other.ncols);
    BaseArray<T>::size = std::move(other.size);
    nb_not_zeros = std::move(other.nb_not_zeros);

    values_ptr = std::exchange(other.values_ptr, nullptr);
    col_index_ptr = std::exchange(other.col_index_ptr, nullptr);
    row_cum_ptr = std::exchange(other.row_cum_ptr, nullptr);
    return *this;
}

template<typename T>
SparseArray<T>::~SparseArray()
{
    delete[] values_ptr;
    delete[] col_index_ptr;
    delete[] row_cum_ptr;
}

template<typename T>
void SparseArray<T>::fill(SparseArray<T>& sparse_array)
{
    for(unsigned element = 0; element < nb_not_zeros; ++element)
    {
        values_ptr[element] = sparse_array.values_ptr[element];
        col_index_ptr[element] = sparse_array.col_index_ptr[element];
    }
    for(unsigned row = 0; row < BaseArray<T>::nchannels * BaseArray<T>::nrows +1; ++row)
        row_cum_ptr[row] = sparse_array.row_cum_ptr[row]; 
}
template<typename T>
void SparseArray<T>::fill(const SparseArray<T>& sparse_array)
{
    for(unsigned element = 0; element < nb_not_zeros; ++element)
    {
        values_ptr[element] = sparse_array.values_ptr[element];
        col_index_ptr[element] = sparse_array.col_index_ptr[element];
    }
    for(unsigned row = 0; row < BaseArray<T>::nchannels * BaseArray<T>::nrows +1; ++row)
        row_cum_ptr[row] = sparse_array.row_cum_ptr[row]; 
}

template<typename T>
void SparseArray<T>::fill(Array<T>& array, T tol)
{
    unsigned count = 0;
    unsigned loc_count;
    for(unsigned channel = 0; channel < BaseArray<T>::nchannels; ++channel){
        for(unsigned row = 0; row < BaseArray<T>::nrows; ++row){
            loc_count = 0;
            for(unsigned col = 0; col < BaseArray<T>::ncols; ++col){
                if( std::abs(array(channel, row, col)) > tol )
                {
                    values_ptr[count] = array(channel, row, col);
                    col_index_ptr[count] = col;
                    ++count;
                    ++loc_count;
                }
            }
            row_cum_ptr[channel * BaseArray<T>::nrows + row +1] = row_cum_ptr[channel * BaseArray<T>::nrows + row] + loc_count;
        }
    }
}

template<typename T>
bool SparseArray<T>::operator==(T value) const
{
    for(unsigned element = 0; element < nb_not_zeros; ++element)
    {
        if( notZero(values_ptr[element] -value) ) return false;
    }
    return true;
}
template<typename T>
bool SparseArray<T>::operator!=(T value) const
{
    for(unsigned element = 0; element < nb_not_zeros; ++element)
    {
        if( isZero(values_ptr[element] -value) ) return false;
    }
    return true;
}
template<typename T>
bool SparseArray<T>::operator==(SparseArray<T>& other) const
{
    if( BaseArray<T>::nchannels != other.nchannels ) return false;
    if( BaseArray<T>::nrows != other.nrows ) return false;
    if( BaseArray<T>::ncols != other.ncols ) return false;
    if( nb_not_zeros != other.nb_not_zeros ) return false;

    for(unsigned element = 0; element < nb_not_zeros; ++element)
    {
        if( notZero(values_ptr[element] -other.values_ptr[element]) ||
                col_index_ptr[element] != other.col_index_ptr[element] ||
                row_cum_ptr[element] != other.row_cum_ptr[element])
        {
             return false;
        }
    }
    return true;
}
template<typename T>
bool SparseArray<T>::operator!=(SparseArray<T>& other) const
{
    if( BaseArray<T>::nchannels != other.nchannels ) return true;
    if( BaseArray<T>::nrows != other.nrows ) return true;
    if( BaseArray<T>::ncols != other.ncols ) return true;

    for(unsigned element = 0; element < nb_not_zeros; ++element)
    {
        if( isZero(values_ptr[element] -other.values_ptr[element]) &&
                col_index_ptr[element] == other.col_index_ptr[element] &&
                row_cum_ptr[element] == other.row_cum_ptr[element])
        {
            return false;
        }
    }
    return true;
}
template<typename T>
Array<T> SparseArray<T>::mult(Array<T>& input_array)
{
    assert(BaseArray<T>::nchannels == 1 && "The array is a tensor. The sparse tensor product is not implemented.");
    assert(BaseArray<T>::ncols == input_array.nrows);
    
    if (input_array.ncols == 1)
        std::cout << "Warning: The input array is a column vector. Use mult_vec instead." << std::endl;
    

    Array<T> output_array(BaseArray<T>::nrows, input_array.ncols);
    #pragma omp parallel
    {
        unsigned row = omp_get_thread_num();
        unsigned nbthreads = omp_get_num_threads();
        
        T sum;
        unsigned k;
        unsigned column;
        unsigned next_row_output;
        unsigned loc_input_cols =  input_array.ncols;
        for(; row < output_array.nrows; row += nbthreads){
            next_row_output = row * output_array.ncols;
            for(column = 0; column < output_array.ncols; ++column){
                sum = (T)0;
                for(k = row_cum_ptr[row]; k < row_cum_ptr[row +1]; ++k){
                    
                    sum += values_ptr[k] * input_array.values_ptr[col_index_ptr[k] *loc_input_cols +column];
                }
                output_array.values_ptr[next_row_output + column] = sum;
            }
        }
    }
    return output_array;
}
template<typename T>
Array<T> SparseArray<T>::mult_vec(Array<T>& input_array)
{
    assert(BaseArray<T>::nchannels == 1 && "The array is a tensor. The sparse tensor product is not implemented.");
    assert(BaseArray<T>::ncols == input_array.nrows);
    assert(input_array.ncols == 1 && "The array must be a column vector");

    Array<T> output_array(BaseArray<T>::nrows, 1);
    #pragma omp parallel
    {
        unsigned row = omp_get_thread_num();
        unsigned nbthreads = omp_get_num_threads();
        
        T sum;
        unsigned k;
        for(; row < output_array.nrows; row += nbthreads){
            sum = (T)0;
            for(k = row_cum_ptr[row]; k < row_cum_ptr[row +1]; ++k){
                
                sum += values_ptr[k] * input_array.values_ptr[col_index_ptr[k]];
            }
            output_array.values_ptr[row] = sum;
        }
    }
    return output_array;
}
template<typename T>
Array<T> SparseArray<T>::linear(Array<T>& input_array, Array<T>& bias)
{
    assert(BaseArray<T>::nchannels == 1 && "The array is a tensor. The sparse tensor product is not implemented.");
    assert(BaseArray<T>::ncols == input_array.nrows);
    assert(bias.ncols == 1 && "The bias array must be a vector.");
    
    if (input_array.ncols == 1)
        std::cout << "Warning: The input array is a column vector. Use mult_vec instead." << std::endl;
    

    Array<T> output_array(BaseArray<T>::nrows, input_array.ncols);
    #pragma omp parallel
    {
        unsigned row = omp_get_thread_num();
        unsigned nbthreads = omp_get_num_threads();
    
        T sum;
        unsigned k;
        unsigned column;
        unsigned next_row_output;
        unsigned loc_input_cols =  input_array.ncols;
        for(; row < output_array.nrows; row += nbthreads){
            next_row_output = row * output_array.ncols;
            for(column = 0; column < output_array.ncols; ++column){
                sum = (T)0;
                for(k = row_cum_ptr[row]; k < row_cum_ptr[row +1]; ++k){
                    
                    sum += values_ptr[k] * input_array.values_ptr[col_index_ptr[k] *loc_input_cols +column];
                }
                output_array.values_ptr[next_row_output + column] = sum + bias.values_ptr[row];
            }
        }
    }
    return output_array;
}
template<typename T>
Array<T> SparseArray<T>::linear_vec(Array<T>& input_array, Array<T>& bias)
{
    assert(BaseArray<T>::nchannels == 1 && "The array is a tensor. The sparse tensor product is not implemented.");
    assert(BaseArray<T>::ncols == input_array.nrows);
    assert(bias.ncols == 1 && "The bias array must be a vector.");
    assert(input_array.ncols == 1 && "The input vector must be a column vector");

    Array<T> output_array(BaseArray<T>::nrows, input_array.ncols);
    #pragma omp parallel
    {
        unsigned row = omp_get_thread_num();
        unsigned nbthreads = omp_get_num_threads();

        T sum;
        unsigned k;
        for(; row < output_array.nrows; row += nbthreads){
            sum = (T)0;
            for(k = row_cum_ptr[row]; k < row_cum_ptr[row +1]; ++k){
                
                sum += values_ptr[k] * input_array.values_ptr[col_index_ptr[k]];
            }
            output_array.values_ptr[row] = sum + bias.values_ptr[row];
        }
    }
    return output_array;
}
template<typename T>
inline const T SparseArray<T>::Value(unsigned element) const
{
    assert(element < nb_not_zeros);
    return values_ptr[element];
}
template<typename T>
inline const unsigned SparseArray<T>::colIndex(unsigned element) const
{
    assert(element < nb_not_zeros);
    return col_index_ptr[element];
}
template<typename T>
inline const unsigned SparseArray<T>::rowCum(unsigned channel, unsigned row) const
{
    assert(channel < BaseArray<T>::nchannels && row <= BaseArray<T>::nrows);
    return row_cum_ptr[channel * BaseArray<T>::nrows + row];
}
template<typename T>
inline const unsigned SparseArray<T>::rowCum(unsigned row) const
{
    assert(BaseArray<T>::nchannels == 1);
    assert(row <= BaseArray<T>::nchannels * BaseArray<T>::nrows);
    return row_cum_ptr[row];
}
template<typename T>
Array<T> SparseArray<T>::toDense()
{
    Array<T> dense_array(BaseArray<T>::nchannels, BaseArray<T>::nrows, BaseArray<T>::ncols);
    unsigned k;
    for(unsigned channel=0; channel < BaseArray<T>::nchannels; ++channel)
    {
        for(unsigned row=0; row < BaseArray<T>::nrows; ++row)
        {
            for(k=row_cum_ptr[channel * BaseArray<T>::nrows + row]; k < row_cum_ptr[channel * BaseArray<T>::nrows + row +1]; ++k)
            {
                dense_array(channel, row, col_index_ptr[k]) = values_ptr[k];
            }
        }
    }
    return dense_array;
}

template<typename T>
std::ostream& operator<<(std::ostream& out, SparseArray<T>& sparse_array)
{
    Array<T> array = sparse_array.toDense();
    out << array;
    return out;
}