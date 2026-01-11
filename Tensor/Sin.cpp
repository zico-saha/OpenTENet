#include "Sin.h"
#include "Tensor.h"


double Math::Sin::operator()(double x) const
{
	return std::sin(x);
}

Tensor Math::Sin::f(const Tensor& tensor) const
{
	return Math::Sin().Apply(tensor);
}
