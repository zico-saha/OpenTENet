#include "Atanh.h"
#include "Tensor.h"


double Math::Atanh::operator()(double x) const
{
	if (x <= -1.0 || x >= 1.0)
	{
		throw std::domain_error("[Math] Inverse Hyperbolic Tangent Function failed: inverse hyperbolic tangent is only defined for values in (-1, 1).");
	}

	return std::atanh(x);
}

Tensor Math::Atanh::f(const Tensor& tensor) const
{
	return Math::Atanh().Apply(tensor);
}
