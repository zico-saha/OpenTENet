#include "SELU.h"
#include "ELU.h"


double Activation::SELU::f(double x) const
{
	double elu = Activation::ELU(Activation::SELU::ALPHA).f(x);
	return (Activation::SELU::LAMBDA * elu);
}

double Activation::SELU::df(double x) const
{
	double d_elu = Activation::ELU(Activation::SELU::ALPHA).df(x);
	return (Activation::SELU::LAMBDA * d_elu);
}
