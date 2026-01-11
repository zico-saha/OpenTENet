#include "HardSigmoid.h"


double Activation::HardSigmoid::f(double x) const
{
	return (x <= -3) ? 0 : ((x >= 3) ? 1 : ((x / 6) + 0.5));
}

double Activation::HardSigmoid::df(double x) const
{
	return (x <= -3 || x >= 3) ? 0 : (1 / 6);
}
