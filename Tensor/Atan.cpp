#include "Atan.h"
#include "Tensor.h"


double Math::Atan::operator()(double x) const
{
	return std::atan(x);
}

Tensor Math::Atan::f(const Tensor& tensor) const
{
	return Math::Atan().Apply(tensor);
}
