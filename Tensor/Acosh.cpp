#include "Acosh.h"
#include "Tensor.h"


double Math::Acosh::operator()(double x) const
{
	if (x < 1.0)
	{
		throw std::domain_error("[Math] Inverse Hyperbolic Cosine Function failed: inverse hyperbolic cosine is only defined for values >= 1.");
	}

	return std::acosh(x);
}

Tensor Math::Acosh::f(const Tensor& tensor) const
{
	return Math::Acosh().Apply(tensor);
}
