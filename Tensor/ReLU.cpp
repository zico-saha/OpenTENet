#include "ReLU.h"


double Activation::ReLU::f(double x) const
{
    return (x >= 0.0) ? x : 0.0;
}

double Activation::ReLU::df(double x) const
{
    return (x >= 0.0) ? 1.0 : 0.0;
}
