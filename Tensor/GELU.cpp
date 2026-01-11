#include "GELU.h"


Activation::GELU::GELU(bool approx)
{
    this->approx = approx;
}

double Activation::GELU::f(double x) const
{
    if (this->approx)
    {
        double inner = x + (Activation::GELU::COEFF * x * x * x);
        double t = std::tanh(Activation::GELU::SQRT_2_OVER_PI * inner);

        return 0.5 * x * (1.0 + t);
    }
    else
    {
        if (x > 10.0)
        {
            return x;
        }
        if (x < -10.0)
        {
            return 0.0;
        }
        return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
    }
}

double Activation::GELU::df(double x) const
{
    if (this->approx)
    {
        double inner = x + (Activation::GELU::COEFF * x * x * x);
        double u = Activation::GELU::SQRT_2_OVER_PI * inner;
        double t = std::tanh(u);

        double du = Activation::GELU::SQRT_2_OVER_PI * (1.0 + (3.0 * Activation::GELU::COEFF * x * x));

        return 0.5 * (1.0 + t + (x * (1.0 - (t * t)) * du));
    }
    else
    {
        double erf_term = std::erf(x / std::sqrt(2.0));
        double exp_term = std::exp(-0.5 * x * x);

        double term_1 = 0.5 * (1.0 + erf_term);
        double term_2 = (x * exp_term) / std::sqrt(2.0 * std::numbers::pi);

        return (term_1 + term_2);
    }
}
