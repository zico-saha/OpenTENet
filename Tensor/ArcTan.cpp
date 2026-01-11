#include "ArcTan.h"


double Activation::ArcTan::f(double x) const
{
    return std::atan(x);
}

double Activation::ArcTan::df(double x) const
{
    return 1.0 / (1.0 + (x * x));
}
