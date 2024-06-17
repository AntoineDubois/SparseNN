#include "utils.hxx"
#include <iostream>


int main()
{
    unsigned error = 0;

    double x = 0.0;
    if (notZero(x)) ++error;
    if (!isZero(x)) ++error;

    x = 1.0;
    if (!notZero(x)) ++error;
    if (isZero(x)) ++error;

    return error;
}