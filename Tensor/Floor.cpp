#include "Floor.h"
#include "Tensor.h"


double Math::Floor::operator()(double x) const
{
	return std::floor(x);
}

Tensor Math::Floor::f(const Tensor& tensor) const
{
	return Math::Floor().Apply(tensor);
}
