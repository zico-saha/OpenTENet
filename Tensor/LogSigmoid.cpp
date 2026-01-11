#include "LogSigmoid.h"
#include "Sigmoid.h"


double Activation::LogSigmoid::f(double x) const
{
	return std::log(Activation::Sigmoid().f(x));
}

double Activation::LogSigmoid::df(double x) const
{
	return 1 - Activation::Sigmoid().f(x);
}
