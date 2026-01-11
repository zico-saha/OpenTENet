#include "Softplus.h"
#include "Sigmoid.h"


double Activation::Softplus::f(double x) const
{
    if (x > 20.0)
    {
        return x;
    }
    if (x < -20.0)
    {
        return std::exp(x);
    }
    return std::log(1.0 + std::exp(x));
}

double Activation::Softplus::df(double x) const
{
    if (x > 20.0)
    {
        return 1;
    }
    if (x < -20.0)
    {
        return std::exp(x);
    }
    return Activation::Sigmoid().f(x);
}
