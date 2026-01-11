#include "PReLU.h"


Activation::PReLU::PReLU(double alpha)
{
    if (alpha < 0.0)
    {
        throw std::invalid_argument("[Activation] PReLU failed: alpha must be non-negative.");
    }

    if (!std::isfinite(alpha))
    {
        throw std::invalid_argument("[Activation] PReLU failed: alpha must be finite.");
    }

    this->alpha = alpha;
}

double Activation::PReLU::f(double x) const
{
    return (x >= 0.0) ? x : (this->alpha * x);
}

double Activation::PReLU::df(double x) const
{
    return (x >= 0.0) ? 1.0 : this->alpha;
}
