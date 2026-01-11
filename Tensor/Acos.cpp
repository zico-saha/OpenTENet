#include "Acos.h"
#include "Tensor.h"


double Math::Acos::operator()(double x) const
{
	if (x < -1.0 || x > 1.0)
	{
		throw std::domain_error("[Math] Arc-cosine Function failed: arccosine is only defined for values in [-1, 1].");
	}

	return std::acos(x);
}

Tensor Math::Acos::f(const Tensor& tensor) const
{
	return Math::Acos().Apply(tensor);
}
