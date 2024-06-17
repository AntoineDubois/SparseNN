#include "flatten.hxx"
#include "algebra.hxx"
#include <iostream>


template<typename T>
Flatten<T>::Flatten()
{
    // nothing here
}
template<typename T>
Flatten<T>::~Flatten()
{
    // nothing here
}

template<typename T>
inline Array<T>& Flatten<T>::operator()(Array<T>& array)
{
    array.flatten();
    return array;
}
template<typename T>
Array<T>& Flatten<T>::Forward(Array<T>& array)
{
    array.flatten();
    return array;
}
template<typename T>
inline std::string Flatten<T>::name()
{
    return "Flatten";
}

template<typename T>
std::ostream& operator<<(std::ostream& out, Flatten<T>& layer)
{
    out << "Flatten";
    return out;
}