#include "Mish.h"
#include "Softplus.h"
#include "Swish.h"


double Activation::Mish::f(double x) const
{
	double softplus = Activation::Softplus().f(x);
	return (x * std::tanh(softplus));
}

double Activation::Mish::df(double x) const
{
	double softplus = Activation::Softplus().f(x);
	double swish = Activation::Swish().f(x);
	double t = std::tanh(softplus);

	return t + swish * (1 - (t * t));
}
