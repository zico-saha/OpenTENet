#include "Gaussian.h"


Activation::Gaussian::Gaussian(double center, double std_dev, double scale)
{
    if (std_dev <= 0.0)
    {
        throw std::invalid_argument("[Activation] Gaussian failed: std_dev must be positive.");
    }

    if (!std::isfinite(center) || !std::isfinite(std_dev) || !std::isfinite(scale))
    {
        throw std::invalid_argument("[Activation] Gaussian failed: center, std deviation, scale must be finite.");
    }

    this->center = center;
    this->std_dev = std_dev;
    this->scale = scale;
}

double Activation::Gaussian::f(double x) const
{
    double var = this->std_dev * this->std_dev;
    double diff = (x - this->center);

    double exponent = -(diff * diff) / (2.0 * var);

    return (this->scale * std::exp(exponent));
}

double Activation::Gaussian::df(double x) const
{
    double var = this->std_dev * this->std_dev;
    double diff = (x - this->center);

    double result = -(diff / var) * this->f(x);

    return result;
}
