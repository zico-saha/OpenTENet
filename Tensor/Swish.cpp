#include "Swish.h"
#include "Sigmoid.h"


double Activation::Swish::f(double x) const
{
    return x * Activation::Sigmoid().f(x);
}

double Activation::Swish::df(double x) const
{
    double s = Activation::Sigmoid().f(x);
    return s * (1 + x - (x * s));
}
