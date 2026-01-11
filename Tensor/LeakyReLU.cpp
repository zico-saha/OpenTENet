#include "LeakyReLU.h"


Activation::LeakyReLU::LeakyReLU(double alpha)
{
    if (alpha < 0.0)
    {
        throw std::invalid_argument("[Activation] LeakyReLU failed: alpha must be non-negative.");
    }

    if (!std::isfinite(alpha))
    {
        throw std::invalid_argument("[Activation] LeakyReLU failed: alpha must be finite.");
    }

    this->alpha = alpha;
}

double Activation::LeakyReLU::f(double x) const
{
    return (x >= 0.0) ? x : (this->alpha * x);
}

double Activation::LeakyReLU::df(double x) const
{
    return (x >= 0.0) ? 1.0 : this->alpha;
}
