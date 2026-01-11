#include "SquarePlus.h"


Activation::SquarePlus::SquarePlus(double smoothness)
{
	if (smoothness <= 0.0)
	{
		throw std::invalid_argument("[Activation] SquarePlus failed: smoothness parameter must be positive.");
	}

	this->smoothness = smoothness;
}

double Activation::SquarePlus::f(double x) const
{
	double a = (x * x) + this->smoothness;
	return (x + std::sqrt(a)) / 2.0;
}

double Activation::SquarePlus::df(double x) const
{
	double a = (x * x) + this->smoothness;
	return (1.0 + (x / std::sqrt(a))) / 2.0;
}
