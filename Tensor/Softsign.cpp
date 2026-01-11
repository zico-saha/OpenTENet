#include "Softsign.h"


double Activation::Softsign::f(double x) const
{
	return x / (std::abs(x) + 1.0);
}

double Activation::Softsign::df(double x) const
{
	double d = 1.0 + std::abs(x);
	return 1.0 / (d * d);
}
