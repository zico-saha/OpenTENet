#include "Linear.h"


double Activation::Linear::f(double x) const
{
	return x;
}

double Activation::Linear::df(double x) const
{
	return 1.0;
}
