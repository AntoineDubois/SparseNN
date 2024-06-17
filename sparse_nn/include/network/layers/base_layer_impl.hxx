#include "base_layer.hxx"
#include <ostream>

template<typename T>
BaseLayer<T>::BaseLayer()
{
    // nothing here
}
template<typename T>
BaseLayer<T>::~BaseLayer()
{
    // nothing here
}
template<typename T>
Array<T> BaseLayer<T>::operator()(Array<T>& input_array)
{
    return input_array;
}
template<typename T>
Array<T> BaseLayer<T>::Forward(Array<T>& input_array)
{
    return input_array;
}
template<typename T>
inline std::string BaseLayer<T>::name()
{
    return "BaseLayer";
}
template<typename T>
std::ostream& operator<<(std::ostream& out, BaseLayer<T>& layer)
{
    out << "BaseLayer";
    return out;
}
