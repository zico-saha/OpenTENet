#include "Asinh.h"
#include "Tensor.h"


double Math::Asinh::operator()(double x) const
{
	return std::asinh(x);
}

Tensor Math::Asinh::f(const Tensor& tensor) const
{
	return Math::Asinh().Apply(tensor);
}
