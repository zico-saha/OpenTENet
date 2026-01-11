#include "Sech.h"
#include "Tensor.h"


double Math::Sech::operator()(double x) const
{
	return 1.0 / std::cosh(x);
}

Tensor Math::Sech::f(const Tensor& tensor) const
{
	return Math::Sech().Apply(tensor);
}
