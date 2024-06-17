#include "max.hxx"

#include "algebra.hxx"
#include <iostream>


template<typename T>
Max<T>::Max()
{
    // nothing here
}
template<typename T>
Max<T>::~Max()
{
    // nothing here
}

template<typename T>
inline Array<T>& Max<T>::operator()(Array<T>& array)
{
    T max = array.max();
    array.realloc(1, 1, 1);
    array.values_ptr[0] = max;
    return array;
}
template<typename T>
Array<T>& Max<T>::Forward(Array<T>& array)
{
    T max = array.max();
    array.realloc(1, 1, 1);
    array.values_ptr[0] = max;
    return array;
}
template<typename T>
inline std::string Max<T>::name()
{
    return "Max";
}

template<typename T>
std::ostream& operator<<(std::ostream& out, Max<T>& layer)
{
    out << "Max";
    return out;
}