#include "Asech.h"
#include "Tensor.h"


double Math::Asech::operator()(double x) const
{
	if (x <= 0.0 || x > 1.0)
	{
		throw std::domain_error("[Math] Inverse Hyperbolic Secant Function failed: inverse hyperbolic secant is only defined for values in (0, 1].");
	}

	return std::acosh(1.0 / x);
}

Tensor Math::Asech::f(const Tensor& tensor) const
{
	return Math::Asech().Apply(tensor);
}
