#pragma once
#include <iostream>
#include "base_array.hxx"

template<typename T>
class Array: public BaseArray<T>
{
public:
    T* values_ptr;
private:
    Array(unsigned nchannels, unsigned nrows, unsigned ncols, unsigned size);
public:
    Array(unsigned nchannels, unsigned nrows, unsigned ncols); // tensor constructor
    Array(unsigned nrows, unsigned ncols); // matrix constructor
    Array(unsigned nelements); // vector constructor
    // copy constructors
    Array(Array& other);
    Array(const Array& other);

    // move constructors
    Array(Array&& other);
    Array(const Array&& other);

    ~Array() override;

    // copy and move assignement opperator
    Array<T>& operator=(Array<T>& other) noexcept;
    Array<T>& operator=(Array<T>&& other) noexcept;
    

    inline T& operator()(unsigned channel, unsigned row, unsigned column);
    inline const T operator()(unsigned channel, unsigned row, unsigned column) const;
    inline T& operator()(unsigned row, unsigned column);
    inline const T operator()(unsigned row, unsigned column) const;
    inline T& operator()(unsigned element);
    inline const T operator()(unsigned element) const;  

    bool operator==(T value) const;
    bool operator!=(T value) const;
    bool operator==(Array<T>& other) const;
    bool operator!=(Array<T>& other) const;

    void fill(T value);
    template<typename Generator>
    void fill(Generator& lambda);
    void fill(Array<T>& array);
    void fill(const Array<T>& array);

    void fill(Array<T>& array, unsigned axis, unsigned where);
    void fill(const Array<T>& array, unsigned axis, unsigned where);

    void reshape(unsigned new_nchannels, unsigned new_nrows, unsigned new_ncols);
    void reshape(unsigned new_nrows, unsigned new_ncols);
    inline void flatten(); 
    void realloc(unsigned nchannels, unsigned nrows, unsigned ncols);

    void concatenate(Array<T>& input_array, unsigned axis);
    
    Array<T> mult(Array<T>& input_array);
    Array<T> mult_vec(Array<T>& input_array);
    Array<T> linear(Array<T>& input_1d_array, Array<T>& bias);
    Array<T> linear_vec(Array<T>& input_1d_array, Array<T>& bias);

    T max();
    unsigned argmax();
    T min();
    unsigned argmin();
private:
    void checkaxis(Array<T>& array, unsigned axis);
    void concatenate_along_channel(Array<T>& array);
    void concatenate_along_row(Array<T>& array);
    void concatenate_along_column(Array<T>& array);
};

template<typename T> 
std::ostream& operator<<(std::ostream& out, const Array<T>& array);

#include "array_impl.hxx"