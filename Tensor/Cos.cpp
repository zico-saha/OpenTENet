#include "Cos.h"
#include "Tensor.h"


double Math::Cos::operator()(double x) const
{
	return std::cos(x);
}

Tensor Math::Cos::f(const Tensor& tensor) const
{
	return Math::Cos().Apply(tensor);
}
