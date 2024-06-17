#pragma once
#include "./base_activation_layer.hxx"
#include "algebra.hxx"

template<typename T>
class Max: public BaseActivationLayer<T>
{
private:
    // nothing here
public:
    Max();
    ~Max() override;

    inline Array<T>& operator()(Array<T>& array) override;
    Array<T>& Forward(Array<T>& array) override;

    inline std::string name() override;
};

#include "max_impl.hxx"