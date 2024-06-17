#pragma once
#include "base_layer.hxx"
#include "linear.hxx"
#include "algebra.hxx"
#include <iostream>

template<typename T>
class SparseLinear: public BaseLayer<T>
{
private:
    // nothing here
public:
    SparseArray<T> weights;
    Array<T> bias;

    SparseLinear(Linear<T>& linear_layer);
    ~SparseLinear() override;
    
    Array<T> operator()(Array<T>& input) override;
    Array<T> Forward(Array<T>& input) override;

    inline std::string name() override;
};

#include "sparse_linear_impl.hxx"