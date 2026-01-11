#include "ELU.h"

Activation::ELU::ELU(double alpha)
{
	if (alpha <= 0.0)
	{
		throw std::invalid_argument("[Activation] ELU failed: alpha must be positive.");
	}

	if (!std::isfinite(alpha))
	{
		throw std::invalid_argument("[Activation] ELU failed: alpha must be finite.");
	}

	this->alpha = alpha;
}

double Activation::ELU::f(double x) const
{
	return (x >= 0) ? x : (alpha * (std::exp(x) - 1));
}

double Activation::ELU::df(double x) const
{
	return (x >= 0) ? 1 : (alpha * std::exp(x));
}
