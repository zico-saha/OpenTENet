#include "Sinh.h"
#include "Tensor.h"


double Math::Sinh::operator()(double x) const
{
	return std::sinh(x);
}

Tensor Math::Sinh::f(const Tensor& tensor) const
{
	return Math::Sinh().Apply(tensor);
}
