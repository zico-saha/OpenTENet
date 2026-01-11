#include "SoftShrink.h"


Activation::SoftShrink::SoftShrink(double threshold)
{
    if (threshold < 0.0)
    {
        throw std::invalid_argument("[Activation] SoftShrink failed: threshold must be non-negative.");
    }

    if (!std::isfinite(threshold))
    {
        throw std::invalid_argument("[Activation] SoftShrink failed: threshold must be finite.");
    }

    this->threshold = threshold;
}

double Activation::SoftShrink::f(double x) const
{
    return (x > this->threshold) ? (x - this->threshold) : ((x < -this->threshold) ? (x + this->threshold) : 0.0);
}

double Activation::SoftShrink::df(double x) const
{
    return (x > this->threshold) ? 1.0 : ((x < -this->threshold) ? 1.0 : 0.0);
}
