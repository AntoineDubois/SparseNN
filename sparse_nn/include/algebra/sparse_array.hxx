#pragma once
#include "base_array.hxx"
#include "array.hxx"
#include <limits>

template<typename T>
class SparseArray: public BaseArray<T>
{
public:
    T* values_ptr; 
    unsigned* col_index_ptr;
    unsigned* row_cum_ptr;
    
public:
    unsigned nb_not_zeros;

    // Conversion constructor
    SparseArray(Array<T>& array, T tol = std::numeric_limits<T>::epsilon());
    // copy constructor
    SparseArray(SparseArray<T>& other);
    SparseArray(const SparseArray<T>& other);
    // move constructor
    SparseArray(SparseArray<T>&& other);
    SparseArray(const SparseArray<T>&& other);
    
    ~SparseArray() override;

    SparseArray<T>& operator=(SparseArray<T>& other) noexcept;
    SparseArray<T>& operator=(SparseArray<T>&& other) noexcept;


    bool operator==(T value) const;
    bool operator!=(T value) const;
    bool operator==(SparseArray<T>& other) const;
    bool operator!=(SparseArray<T>& other) const;


    void fill(Array<T>& array, T tol);
    void fill(SparseArray<T>& array);
    void fill(const SparseArray<T>& array);
    
    
    Array<T> mult(Array<T>& input_array);
    Array<T> mult_vec(Array<T>& input_1d_array);
    Array<T> linear(Array<T>& input_array, Array<T>& bias);
    Array<T> linear_vec(Array<T>& input_1d_array, Array<T>& bias);

    inline const T Value(unsigned element) const;
    inline const unsigned colIndex(unsigned element) const;
    inline const unsigned rowCum(unsigned channel, unsigned row) const;
    inline const unsigned rowCum(unsigned row) const;

    Array<T> toDense();

};

template<typename T>
std::ostream& operator<<(std::ostream& out, SparseArray<T>& sparse_array);

#include "sparse_array_impl.hxx"
