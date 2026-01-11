#include "HardShrink.h"


Activation::HardShrink::HardShrink(double threshold)
{
    if (threshold < 0.0)
    {
        throw std::invalid_argument("[Activation] HardShrink failed: threshold must be non-negative.");
    }

    if (!std::isfinite(threshold))
    {
        throw std::invalid_argument("[Activation] HardShrink failed: threshold must be finite.");
    }

    this->threshold = threshold;
}

double Activation::HardShrink::f(double x) const
{
    return (std::fabs(x) > this->threshold) ? x : 0.0;
}

double Activation::HardShrink::df(double x) const
{
    return (std::fabs(x) > this->threshold) ? 1.0 : 0.0;
}
