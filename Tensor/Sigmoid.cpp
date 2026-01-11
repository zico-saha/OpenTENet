#include "Sigmoid.h"


double Activation::Sigmoid::f(double x) const
{
    return 1.0 / (1.0 + std::exp(-x));
}

double Activation::Sigmoid::df(double x) const
{
    double s = this->f(x);
    return s * (1.0 - s);
}
