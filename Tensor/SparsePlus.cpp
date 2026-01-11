#include "SparsePlus.h"


double Activation::SparsePlus::f(double x) const
{
	return (x <= -1.0) ? 0 : ((x >= 1.0) ? x : (0.25 * (x + 1.0) * (x + 1.0)));
}

double Activation::SparsePlus::df(double x) const
{
	return (x <= -1.0) ? 0 : ((x >= 1.0) ? 1 : (0.5 * (x + 1.0)));
}
