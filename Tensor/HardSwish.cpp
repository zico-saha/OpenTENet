#include "HardSwish.h"


double Activation::HardSwish::f(double x) const
{
	return (x <= -3) ? 0 : ((x >= 3) ? x : (x * (x + 3) / 6));
}

double Activation::HardSwish::df(double x) const
{
	return (x <= -3) ? 0 : ((x >= 3) ? 1 : ((x / 3) + 0.5));
}
