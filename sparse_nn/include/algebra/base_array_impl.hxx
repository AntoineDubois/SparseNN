#include "base_array.hxx"
#include <cassert>

template<typename T>
BaseArray<T>::BaseArray(unsigned nchannels, unsigned nrows, unsigned ncols): nchannels(nchannels), nrows(nrows), ncols(ncols), size(nchannels * nrows * ncols)
{
    checkinputs();
}
template<typename T>
BaseArray<T>::~BaseArray()
{
    // nothing here
}
template<typename T>
BaseArray<T>::BaseArray(unsigned nchannels, unsigned nrows, unsigned ncols, unsigned size): nchannels(nchannels), nrows(nrows), ncols(ncols), size(size)
{
    checkinputs();
    assert(size == nchannels * nrows * ncols);
}

template<typename T>
void BaseArray<T>::checkinputs()
{
    assert(nchannels>0 && "An array has a least one channel.");
    assert(nrows>0 && "An array has a least one row");
    assert(ncols>0 && "An array has a least one column");

}