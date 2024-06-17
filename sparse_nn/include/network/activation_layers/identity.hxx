#pragma once
#include "algebra.hxx"
#include "base_activation_layer.hxx"
#include <iostream>

template<typename T>
class Identity: public BaseActivationLayer<T>
{
private:
    //
public:
    Identity();
    ~Identity() override;

    inline Array<T>& operator()(Array<T>& input_array) override;
    inline Array<T>& Forward(Array<T>& input_array) override;

    inline std::string name() override;
};

#include "identity_impl.hxx"