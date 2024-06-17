#pragma once
#include "algebra.hxx"
#include "base_activation_layer.hxx"
#include <iostream>


template<typename T>
class LogSoftmax: public BaseActivationLayer<T>
{
private:
    //
public:
    LogSoftmax();
    ~LogSoftmax() override;

    Array<T>& operator()(Array<T>& input_array) override;
    Array<T>& Forward(Array<T>& input_array) override;

    inline std::string name() override;
};

#include "logsoftmax_impl.hxx"