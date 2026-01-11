#include "Cosh.h"
#include "Tensor.h"


double Math::Cosh::operator()(double x) const
{
	return std::cosh(x);
}

Tensor Math::Cosh::f(const Tensor& tensor) const
{
	return Math::Cosh().Apply(tensor);
}
