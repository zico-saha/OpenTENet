#include "HardTanh.h"


double Activation::HardTanh::f(double x) const
{
	return (x <= -1) ? -1 : ((x >= 1) ? 1 : x);
}

double Activation::HardTanh::df(double x) const
{
	return (x < -1 || x > 1) ? 0 : 1;
}
