#include "identity.hxx"
#include <iostream>

template<typename T>
Identity<T>::Identity()
{
    // nothing here
}
template<typename T>
Identity<T>::~Identity()
{
    // nothing here
}

template<typename T>
inline Array<T>& Identity<T>::operator()(Array<T>& input_array)
{
    return input_array;
}
template<typename T>
inline Array<T>& Identity<T>::Forward(Array<T>& input_array)
{
    return input_array;
}
template<typename T>
inline std::string Identity<T>::name()
{
    return "Identity";
}
template<typename T>
std::ostream& operator<<(std::ostream& out, Identity<T>& layer)
{
    out << "Identity";
    return out;
}