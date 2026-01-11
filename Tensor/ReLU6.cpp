#include "ReLU6.h"


double Activation::ReLU6::f(double x) const
{
    return (x <= 0.0) ? 0.0 : ((x >= 6.0) ? 6.0 : x);
}

double Activation::ReLU6::df(double x) const
{
    return (x <= 0.0 || x >= 6.0) ? 0.0 : 1.0;
}
