#include "Exponential.h"


double Activation::Exponential::f(double x) const
{
    if (x > Activation::Exponential::X_LIMIT)
    {
        throw std::overflow_error("[Activation] Exponential failed: input too large, would cause overflow.");
    }
    return std::exp(x);
}

double Activation::Exponential::df(double x) const
{
    return this->f(x);
}
