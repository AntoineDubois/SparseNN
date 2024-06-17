#pragma once

template<typename T>
class BaseArray
{
public:
    unsigned size;
    unsigned nchannels;
    unsigned ncols;
    unsigned nrows;
    BaseArray(unsigned nchannels, unsigned nrows, unsigned ncols);
    BaseArray(unsigned nchannels, unsigned nrows, unsigned ncols, unsigned size);
    virtual ~BaseArray();
protected:
    void checkinputs();
};

#include "base_array_impl.hxx"