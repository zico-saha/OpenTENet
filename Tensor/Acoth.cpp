#include "Acoth.h"
#include "Tensor.h"


double Math::Acoth::operator()(double x) const
{
	if (x >= -1.0 && x <= 1.0)
	{
		throw std::domain_error("[Math] Inverse Hyperbolic Cotangent Function failed: inverse hyperbolic cotangent is only defined for values in (-inf, -1) ∪ (1, inf).");
	}

	return std::atanh(1.0 / x);
}

Tensor Math::Acoth::f(const Tensor& tensor) const
{
	return Math::Acoth().Apply(tensor);
}
