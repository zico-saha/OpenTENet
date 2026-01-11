#include "_Tanh.h"


double Activation::Tanh::f(double x) const
{
    return std::tanh(x);
}

double Activation::Tanh::df(double x) const
{
    double t = std::tanh(x);
    return 1.0 - (t * t);
}
