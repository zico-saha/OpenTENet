#include "Ceil.h"
#include "Tensor.h"


double Math::Ceil::operator()(double x) const
{
	return std::ceil(x);
}

Tensor Math::Ceil::f(const Tensor& tensor) const
{
	return Math::Ceil().Apply(tensor);
}
