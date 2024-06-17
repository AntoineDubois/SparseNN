#include "array.hxx"
#include "utils.hxx"
#include <cassert>
#include <omp.h>

#include <iostream>

template<typename T>
Array<T>::Array(unsigned nchannels, unsigned nrows, unsigned ncols, unsigned size): BaseArray<T>(nchannels, nrows, ncols, size)
{
    values_ptr = new T[BaseArray<T>::size];
}
template<typename T>
Array<T>::Array(unsigned nchannels, unsigned nrows, unsigned ncols): BaseArray<T>(nchannels, nrows, ncols)
{
    values_ptr = new T[BaseArray<T>::size];
}
template<typename T>
Array<T>::Array(unsigned nrows, unsigned ncols): Array<T>::Array(1, nrows, ncols)
{
    // nothing here
}
template<typename T>
Array<T>::Array(unsigned nelements): Array<T>::Array(1, nelements, 1)
{
    // nothing here
}
template<typename T> 
Array<T>::Array(Array& other): BaseArray<T>(other.nchannels, other.nrows, other.ncols, other.size)
{
    values_ptr = new T[BaseArray<T>::size];
    fill(other);
}
template<typename T> 
Array<T>::Array(const Array<T>& other):  BaseArray<T>(other.nchannels, other.nrows, other.ncols, other.size)
{
    values_ptr = new T[BaseArray<T>::size];
    fill(other);
}
template<typename T>
Array<T>::Array(Array&& other): BaseArray<T>(std::move(other.nchannels), std::move(other.nrows), std::move(other.ncols), std::move(other.size))
{
    values_ptr = std::exchange(other.values_ptr, nullptr);
}
template<typename T>
Array<T>::Array(const Array&& other): BaseArray<T>(std::move(other.nchannels), std::move(other.nrows), std::move(other.ncols), std::move(other.size))
{
    values_ptr = std::exchange(other.values_ptr, nullptr);
}
template<typename T>
Array<T>::~Array()
{
    delete[] values_ptr;
}
template<typename T>
Array<T>& Array<T>::operator=(Array<T>& other) noexcept
{
    if ( this == &other ) return *this;

    delete[] values_ptr;
    BaseArray<T>::nchannels = other.nchannels;
    BaseArray<T>::nrows = other.nrows;
    BaseArray<T>::ncols = other.ncols;
    BaseArray<T>::size = other.size;
    values_ptr = new T[BaseArray<T>::size];
    std::copy(other.values_ptr, other.values_ptr +BaseArray<T>::size, values_ptr);

    return *this;
}
template<typename T>
Array<T>& Array<T>::operator=(Array<T>&& other) noexcept
{
    if ( this == &other ) return *this;

    delete[] values_ptr;
    BaseArray<T>::nchannels = std::move(other.nchannels);
    BaseArray<T>::nrows = std::move(other.nrows);
    BaseArray<T>::ncols = std::move(other.ncols);
    BaseArray<T>::size = std::move(other.size);
    values_ptr = std::exchange(other.values_ptr, nullptr);

    return *this;
}

template<typename T>
inline T& Array<T>::operator()(unsigned channel, unsigned row, unsigned column)
{
    assert(channel < BaseArray<T>::nchannels && row < BaseArray<T>::nrows && column < BaseArray<T>::ncols);
    return values_ptr[(channel * BaseArray<T>::nrows + row) * BaseArray<T>::ncols + column];
}
template<typename T>
inline const T Array<T>::operator()(unsigned channel, unsigned row, unsigned column) const
{
    assert(channel < BaseArray<T>::nchannels && row < BaseArray<T>::nrows && column < BaseArray<T>::ncols);
    return values_ptr[(channel * BaseArray<T>::nrows + row) * BaseArray<T>::ncols + column];
}
template<typename T>
inline T& Array<T>::operator()(unsigned row, unsigned column)
{
    assert(BaseArray<T>::nchannels == 1);
    assert(row < BaseArray<T>::nrows && column < BaseArray<T>::ncols);
    return values_ptr[row * BaseArray<T>::ncols + column];
}
template<typename T>
inline const T Array<T>::operator()(unsigned row, unsigned column) const
{
    assert(BaseArray<T>::nchannels == 1);
    assert(row < BaseArray<T>::nrows && column < BaseArray<T>::ncols);
    return values_ptr[row * BaseArray<T>::ncols + column];
}
template<typename T>
inline T& Array<T>::operator()(unsigned element)
{
    assert( element < BaseArray<T>::size);
    return values_ptr[element];
}
template<typename T>
inline const T Array<T>::operator()(unsigned element) const
{
    assert( element < BaseArray<T>::size );
    return values_ptr[element];
}
template<typename T>
bool Array<T>::operator==(T value) const
{
    for(unsigned channel = 0; channel < BaseArray<T>::nchannels; ++channel){
        for(unsigned row = 0; row < BaseArray<T>::nrows; ++row){
            for(unsigned col = 0; col < BaseArray<T>::ncols; ++col){
                if ( notZero( operator()(channel, row, col) -value) ) return false;
            }
        }
    }
    return true;
}
template<typename T>
bool Array<T>::operator!=(T value) const
{
    for(unsigned channel = 0; channel < BaseArray<T>::nchannels; ++channel){
        for(unsigned row = 0; row < BaseArray<T>::nrows; ++row){
            for(unsigned col = 0; col < BaseArray<T>::ncols; ++col){
                if ( isZero(operator()(channel, row, col) -value) ) return false;
            }
        }
    }
    return true;
}
template<typename T>
bool Array<T>::operator==(Array<T>& other) const
{
    if( BaseArray<T>::nchannels != other.nchannels ) return false;
    if( BaseArray<T>::nrows != other.nrows ) return false;
    if( BaseArray<T>::ncols != other.ncols ) return false;

    for(unsigned channel = 0; channel < BaseArray<T>::nchannels; ++channel){
        for(unsigned row = 0; row < BaseArray<T>::nrows; ++row){
            for(unsigned col = 0; col < BaseArray<T>::ncols; ++col){
                if ( notZero(operator()(channel, row, col) -other(channel, row, col)) ) return false;
            }
        }
    }
    return true;
}
template<typename T>
bool Array<T>::operator!=(Array<T>& other) const
{
    if( BaseArray<T>::nchannels != other.nchannels ) return true;
    if( BaseArray<T>::nrows != other.nrows ) return true;
    if( BaseArray<T>::ncols != other.ncols ) return true;

    for(unsigned channel = 0; channel < BaseArray<T>::nchannels; ++channel){
        for(unsigned row = 0; row < BaseArray<T>::nrows; ++row){
            for(unsigned col = 0; col < BaseArray<T>::ncols; ++col){
                if ( isZero(operator()(channel, row, col) -other(channel, row, col)) ) return false;
            }
        }
    }
    return true;
}
template<typename T>
void Array<T>::fill(T value)
{
    for (unsigned element = 0; element < BaseArray<T>::size; ++element)
        values_ptr[element] = value;
}
template<typename T> 
template<typename Generator>
void Array<T>::fill(Generator& lambda)
{
    for (unsigned element = 0; element < BaseArray<T>::size; ++element)
        values_ptr[element] = lambda();
}
template<typename T>
void Array<T>::fill(Array<T>& array)
{
    for(unsigned element = 0; element < BaseArray<T>::size; ++element)
        values_ptr[element] = array.values_ptr[element];
}
template<typename T>
void Array<T>::fill(const Array<T>& array)
{
    for(unsigned element = 0; element < BaseArray<T>::size; ++element)
        values_ptr[element] = array.values_ptr[element];
}
template<typename T>
void Array<T>::fill(Array<T>& array, unsigned axis, unsigned where)
{
    assert(axis == 0 || axis == 1 || axis == 2);
    if (axis == 0)
    {
        assert(array.nchannels == 1);
        assert(array.nrows = BaseArray<T>::nrows);
        assert(array.ncols = BaseArray<T>::ncols);
        assert(where < BaseArray<T>::nchannels);
        for(unsigned row = 0; row < BaseArray<T>::nrows; ++row){
            for(unsigned column = 0; column < BaseArray<T>::ncols; ++column){
                operator()(where, row, column) = array(0, row, column);
            }    
        }
        
    }
    else if (axis == 1)
    {
        assert(array.nchannels == BaseArray<T>::nchannels);
        assert(array.nrows = 1);
        assert(array.ncols = BaseArray<T>::ncols);
        assert(where < BaseArray<T>::nrows);
        for(unsigned channel = 0; channel < BaseArray<T>::nchannels; ++channel){
            for(unsigned column = 0; column < BaseArray<T>::ncols; ++column){
                operator()(channel, where, column) = array(channel, 0, column);
            }    
        }
    }
    else // if (axis == 2)
    {
        assert(array.nchannels == BaseArray<T>::nchannels);
        assert(array.nrows = BaseArray<T>::nrows);
        assert(array.ncols = 1);
        assert(where < BaseArray<T>::nrows);
        for(unsigned channel = 0; channel < BaseArray<T>::nchannels; ++channel){
            for(unsigned row = 0; row < BaseArray<T>::nrows; ++row){
                operator()(channel, row, where) = array(channel, row, 0);
            }    
        }
    }
}
template<typename T>
void Array<T>::fill(const Array<T>& array, unsigned axis, unsigned where)
{
    assert(axis == 0 || axis == 1 || axis == 2);
    if (axis == 0)
    {
        assert(array.nchannels == 1);
        assert(array.nrows = BaseArray<T>::nrows);
        assert(array.ncols = BaseArray<T>::ncols);
        assert(where < BaseArray<T>::nchannels);
        for(unsigned row = 0; row < BaseArray<T>::nrows; ++row){
            for(unsigned column = 0; column < BaseArray<T>::ncols; ++column){
                operator()(where, row, column) = array(0, row, column);
            }    
        }
        
    }
    else if (axis == 1)
    {
        assert(array.nchannels == BaseArray<T>::nchannels);
        assert(array.nrows = 1);
        assert(array.ncols = BaseArray<T>::ncols);
        assert(where < BaseArray<T>::nrows);
        for(unsigned channel = 0; channel < BaseArray<T>::nchannels; ++channel){
            for(unsigned column = 0; column < BaseArray<T>::ncols; ++column){
                operator()(channel, where, column) = array(channel, 0, column);
            }    
        }
    }
    else // if (axis == 2)
    {
        assert(array.nchannels == BaseArray<T>::nchannels);
        assert(array.nrows = BaseArray<T>::nrows);
        assert(array.ncols = 1);
        assert(where < BaseArray<T>::nrows);
        for(unsigned channel = 0; channel < BaseArray<T>::nchannels; ++channel){
            for(unsigned row = 0; row < BaseArray<T>::nrows; ++row){
                operator()(channel, row, where) = array(channel, row, 0);
            }    
        }
    }
}
template<typename T>
void Array<T>::reshape(unsigned new_nchannels, unsigned new_nrows, unsigned new_ncols)
{
    assert(new_nchannels * new_nrows * new_ncols == BaseArray<T>::size && "The number of element cannot change.");

    BaseArray<T>::nchannels = new_nchannels;
    BaseArray<T>::nrows = new_nrows;
    BaseArray<T>::ncols = new_ncols;
}
template<typename T>
void Array<T>::reshape(unsigned new_nrows, unsigned new_ncols)
{
    assert(BaseArray<T>::nchannels == 1 && "The array is a tensor. Use reshape(nchannels, nrows, ncols) instead.");
    assert(new_nrows * new_ncols == BaseArray<T>::nrows * BaseArray<T>::ncols);
    reshape(BaseArray<T>::nchannels, new_nrows, new_ncols);
}
template<typename T>
inline void Array<T>::flatten()
{
    BaseArray<T>::nchannels = 1;
    BaseArray<T>::nrows = BaseArray<T>::size;
    BaseArray<T>::ncols = 1;
}
template<typename T>
void Array<T>::realloc(unsigned nchannels, unsigned nrows, unsigned ncols)
{
    BaseArray<T>::nchannels = nchannels;
    BaseArray<T>::nrows = nrows;
    BaseArray<T>:: ncols = ncols;
    BaseArray<T>::size = nchannels * nrows * ncols;
    
    T* new_ptr = new T[BaseArray<T>::size];
    delete[] values_ptr;
    values_ptr = std::exchange(new_ptr, nullptr);
}
template<typename T>
void Array<T>::concatenate(Array<T>& array, unsigned axis)
{
    checkaxis(array, axis);

    if (axis == 0) {
        concatenate_along_channel(array);
    } else if (axis == 1) {
        concatenate_along_row(array);
    } else if (axis == 2) {
        concatenate_along_column(array);
    }
    BaseArray<T>::size = BaseArray<T>::size + array.size;
}
template<typename T>
Array<T> Array<T>::mult(Array<T>& input_array)
{
    assert(BaseArray<T>::nchannels == 1 && "The array cannot be a tensor. The tensor product is not implemented.");
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
        unsigned next_row;
        for(; row < output_array.nrows; row += nbthreads){
            next_row = row * BaseArray<T>::ncols;
            for(column = 0; column < output_array.ncols; ++column){
                sum = (T)0;
                for(k = 0; k < BaseArray<T>::ncols; ++k){
                    sum += values_ptr[next_row +k] * input_array.values_ptr[k * input_array.ncols +column];
                }
            output_array.values_ptr[row * output_array.ncols + column] = sum;
            }   
        }
    }
    return output_array;
}
template<typename T>
Array<T> Array<T>::mult_vec(Array<T>& input_array)
{
    assert(BaseArray<T>::nchannels == 1 && "The array cannot be a tensor. The tensor product is not implemented.");
    assert(BaseArray<T>::ncols == input_array.nrows);
    assert(input_array.ncols == 1 && "The input array must be a column vector.");

    Array<T> output_array(BaseArray<T>::nrows, 1);
    #pragma omp parallel
    {
        unsigned row = omp_get_thread_num();
        unsigned nbthreads = omp_get_num_threads();
        
        T sum;
        unsigned k;
        unsigned next_row;
        for(; row < output_array.nrows; row += nbthreads){
            sum = (T)0;
            next_row = row * BaseArray<T>::ncols;            
            for(k = 0; k < BaseArray<T>::ncols; ++k){
                sum += values_ptr[next_row +k] * input_array.values_ptr[k];
            }
            output_array.values_ptr[row] = sum;
        }
    }
    return output_array;
}
template<typename T>
Array<T> Array<T>::linear(Array<T>& input_array, Array<T>& bias)
{
    assert(BaseArray<T>::nchannels == 1 && "The array is a tensor. The tensor product is not implemented.");
    assert(BaseArray<T>::ncols == input_array.nrows);
    assert(bias.ncols == 1 && "The bias array must be a column vector.");
    
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
        unsigned next_row;
        for(; row < output_array.nrows; row += nbthreads){
            next_row = row * BaseArray<T>::ncols;
            for(column = 0; column < output_array.ncols; ++column){
                sum = (T)0;
                for(k = 0; k < BaseArray<T>::ncols; ++k){
                    sum += values_ptr[next_row +k] * input_array.values_ptr[k * input_array.ncols +column];
                }
                output_array.values_ptr[row * output_array.ncols + column] = sum + bias.values_ptr[row];
            }
        }
    }
    return output_array;
}
template<typename T>
Array<T> Array<T>::linear_vec(Array<T>& input_array, Array<T>& bias)
{
    assert(BaseArray<T>::nchannels == 1 && "The array is a tensor. The tensor product is not implemented.");
    assert(BaseArray<T>::ncols == input_array.nrows);
    assert(bias.ncols == 1 && "The bias array must be a column vector.");
    assert(input_array.ncols == 1 && "The input array must be a column vector");

    Array<T> output_array(BaseArray<T>::nrows, input_array.ncols);
    #pragma omp parallel
    {
        unsigned row = omp_get_thread_num();
        unsigned nbthreads = omp_get_num_threads();

        T sum;
        unsigned k;
        unsigned next_row;
        for(; row < output_array.nrows; row += nbthreads){
            sum = (T)0;
            next_row = row * BaseArray<T>::ncols;            
            for(k = 0; k < BaseArray<T>::ncols; ++k){
                sum += values_ptr[next_row +k] * input_array.values_ptr[k];
            }
            output_array.values_ptr[row] = sum + bias.values_ptr[row];
        }
    }
    return output_array;
}
template<typename T>
T Array<T>::max()
{
    T max_element = values_ptr[0];
    for(unsigned element=1; element < BaseArray<T>::size; ++element)
    {
        if(values_ptr[element] > max_element) max_element = values_ptr[element];
    }
    return max_element;
}
template<typename T>
unsigned Array<T>::argmax()
{
    T max_element = values_ptr[0];
    unsigned max_index = 0;
    for(unsigned element=1; element < BaseArray<T>::size; ++element)
    {
        if(values_ptr[element] > max_element){
            max_element = values_ptr[element];
            max_index = element;
        }
    }
    return max_index;
}
template<typename T>
T Array<T>::min()
{
    T min_element = values_ptr[0];
    for(unsigned element=1; element < BaseArray<T>::size; ++element)
    {
        if(values_ptr[element] < min_element) min_element = values_ptr[element];
    }
    return min_element;
}
template<typename T>
unsigned Array<T>::argmin()
{
    T min_element = values_ptr[0];
    unsigned min_index = 0;
    for(unsigned element=1; element < BaseArray<T>::size; ++element)
    {
        if(values_ptr[element] < min_element){
            min_element = values_ptr[element];
            min_index = element;
        }
    }
    return min_index;
}
template<typename T>
void Array<T>::checkaxis(Array<T>& array, unsigned axis)
{
    if ( axis == 0 ) {
        assert( array.nrows == BaseArray<T>::nrows && "The arrays must have the same number of rows.");
        assert( array.ncols == BaseArray<T>::ncols && "The arrays must have the same number of columns.");
    } else if ( axis == 1 ) {
        assert( array.nchannels == BaseArray<T>::nchannels && "The arrays must have the same number of channels.");
        assert( array.ncols == BaseArray<T>::ncols && "The arrays must have the same number of columns.");
    } else if ( axis == 2 ) {
        assert( array.nchannels == BaseArray<T>::nchannels && "The arrays must have the same number of channels.");
        assert( array.nrows == BaseArray<T>::nrows && "The arrays must have the same number of rows.");
    } else {
        assert( axis < 3 && "An array has only 3 axes.");
    }
}
template<typename T>
void Array<T>::concatenate_along_channel(Array<T>& array)
{
    T* temp = new T[BaseArray<T>::size + array.size];
    
    for (unsigned element = 0; element < BaseArray<T>::size; ++element )
        temp[element] = values_ptr[element];
    for (unsigned element = 0; element < array.size; ++element )
        temp[BaseArray<T>::size + element] = array.values_ptr[element];

    BaseArray<T>::nchannels = BaseArray<T>::nchannels + array.nchannels;
    
    delete[] values_ptr;
    values_ptr = temp;
}
template<typename T>
void Array<T>::concatenate_along_row(Array<T>& array)
{
    T* temp = new T[BaseArray<T>::size + array.size];

    for (unsigned element = 0; element < BaseArray<T>::size; ++element )
        temp[element] = values_ptr[element];
    for (unsigned element = 0; element < array.size; ++element )
        temp[BaseArray<T>::size + element] = array.values_ptr[element];

    BaseArray<T>::nrows = BaseArray<T>::nrows + array.nrows;

    delete[] values_ptr;
    values_ptr = temp;
}
template<typename T>
void Array<T>::concatenate_along_column(Array<T>& array)
{
    T* temp = new T[BaseArray<T>::size + array.size];
    
    for(unsigned channel = 0; channel < BaseArray<T>::nchannels; ++channel)
    {
        for (unsigned row = 0; row < BaseArray<T>::nrows; ++row)
        {
            for (unsigned element = 0; element < BaseArray<T>::ncols; ++element)
                temp[ channel * BaseArray<T>::nrows * BaseArray<T>::ncols + row * (BaseArray<T>::ncols + array.ncols) + element] = values_ptr[channel * BaseArray<T>::nrows + row * BaseArray<T>::ncols + element];
            for (unsigned element = 0; element < BaseArray<T>::ncols; ++element)
                temp[ channel * BaseArray<T>::nrows * BaseArray<T>::ncols + row * (BaseArray<T>::ncols + array.ncols) + BaseArray<T>::ncols + element] = array.values_ptr[channel * BaseArray<T>::nrows + row * BaseArray<T>::ncols + element];   
        }
    }
    BaseArray<T>::ncols = BaseArray<T>::ncols + array.ncols;
    
    delete[] values_ptr;
    values_ptr = temp;
}


template<typename T> 
std::ostream& operator<<(std::ostream& out, const Array<T>& array)
{
    out << "\n\n";
    for(unsigned channel=0; channel < array.nchannels; ++channel){
        for(unsigned row=0; row < array.nrows; ++row){
            for(unsigned column=0; column < array.ncols; ++column){
                out << array(channel, row, column) << "\t";
            }
            out << "\n";
        }
        out << "\n";
    }
    return out;
}
