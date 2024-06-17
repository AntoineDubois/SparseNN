#include "linear.hxx"
#include "algebra.hxx"
#include <iostream>

template<typename T>
Linear<T>::Linear(unsigned input_size, unsigned output_size): weights(output_size, input_size), bias(output_size)
{
    // nothing here
}
template<typename T>
Linear<T>::Linear(unsigned input_size, unsigned output_size, T init_value_weights, T init_value_bias): Linear(input_size, output_size)
{
    weights.fill(init_value_weights);
    bias.fill(init_value_bias);
}
template<typename T>
template<typename Generator>
Linear<T>::Linear(unsigned input_size, unsigned output_size, Generator && lambda_weights, T init_value_bias): Linear(input_size, output_size)
{
    weights.fill(lambda_weights);
    bias.fill(init_value_bias);
}
template<typename T>
template<typename Generator>
Linear<T>::Linear(unsigned input_size, unsigned output_size, T init_value_weights, Generator && lambda_bias): Linear(input_size, output_size)
{
    weights.fill(init_value_weights);
    bias.fill(lambda_bias);
}
template<typename T>
template<typename Generator>
Linear<T>::Linear(unsigned input_size, unsigned output_size, Generator && lambda_weights, Generator && lambda_bias): Linear(input_size, output_size)
{
    weights.fill(lambda_weights);
    bias.fill(lambda_bias);
}

template<typename T>
Linear<T>::~Linear()
{
    // nothing here
}

template<typename T>
Array<T> Linear<T>::operator()(Array<T>& input)
{
    assert(input.nrows == weights.ncols);
    if (input.ncols > 1)
        return weights.linear(input, bias);
    return weights.linear_vec(input, bias);
}
template<typename T>
Array<T> Linear<T>::Forward(Array<T>& input)
{
    assert(input.nrows == weights.ncols);
    if (input.ncols > 1)
        return weights.linear(input, bias);
    return weights.linear_vec(input, bias);
}
template<typename T>
inline std::string Linear<T>::name()
{
    return "Linear";
}
template<typename T>
std::ostream& operator<<(std::ostream& out, Linear<T>& linear)
{
    out << "Linear(input=" << linear.weights.ncols << ", output=" << linear.weights.nrows;
    return out;
}