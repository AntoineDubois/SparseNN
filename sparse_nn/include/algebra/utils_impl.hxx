#include "utils.hxx"
#include <limits>
#include <cmath>


template<typename T>
inline bool notZero(T x) // relative tolerance test
{
    if (std::abs(x) > std::numeric_limits<T>::epsilon())
        return true;
    
    return false;
}

template<typename T>
inline bool isZero(T x) // relative tolerance test
{
    if (std::abs(x) > std::numeric_limits<T>::epsilon())
        return false;
    
    return true;
}
