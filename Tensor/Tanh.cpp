#include "Tanh.h"
#include "Tensor.h"


double Math::Tanh::operator()(double x) const
{
	return std::tanh(x);
}

Tensor Math::Tanh::f(const Tensor& tensor) const
{
	return Math::Tanh().Apply(tensor);
}
