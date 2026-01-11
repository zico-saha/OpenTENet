#include "TanhShrink.h"


double Activation::TanhShrink::f(double x) const
{
	return x - std::tanh(x);
}

double Activation::TanhShrink::df(double x) const
{
	return std::tanh(x) * std::tanh(x);
}
