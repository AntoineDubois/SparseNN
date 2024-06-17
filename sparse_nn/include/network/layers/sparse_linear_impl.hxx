#include "linear.hxx"
#include "algebra.hxx"
#include <iostream>

template<typename T>
SparseLinear<T>::SparseLinear(Linear<T>& linear_layer): weights(linear_layer.weights), bias(linear_layer.bias)
{
    // nothing here
}
template<typename T>
SparseLinear<T>::~SparseLinear()
{
    // nothing here
}

template<typename T>
Array<T> SparseLinear<T>::operator()(Array<T>& input)
{
    assert(input.nrows == weights.ncols);
    if (input.ncols > 1)
        return weights.linear(input, bias);
    return weights.linear_vec(input, bias);
}
template<typename T>
Array<T> SparseLinear<T>::Forward(Array<T>& input)
{
    assert(input.nrows == weights.ncols);
    if (input.ncols > 1)
        return weights.linear(input, bias);
    return weights.linear_vec(input, bias);
}
template<typename T>
inline std::string SparseLinear<T>::name()
{
    return "SparseLinear";
}
template<typename T>
std::ostream& operator<<(std::ostream& out, SparseLinear<T>& linear)
{
    out << "SparseLinear(input=" << linear.weights.ncols << ", output=" << linear.weights.nrows;
    return out;
}