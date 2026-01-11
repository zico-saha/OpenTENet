#include "Abs.h"
#include "Tensor.h"


double Math::Abs::operator()(double x) const
{
	return std::abs(x);
}

Tensor Math::Abs::f(const Tensor& tensor) const
{
	return Math::Abs().Apply(tensor);
}
