#include "Asec.h"
#include "Tensor.h"


double Math::Asec::operator()(double x) const
{
	if (x > -1.0 && x < 1.0)
	{
		throw std::domain_error("[Math] Arc-secant Function failed: arcsecant is only defined for values in (-inf, -1] ∪ [1, inf).");
	}

	return std::acos(1.0 / x);
}

Tensor Math::Asec::f(const Tensor& tensor) const
{
	return Math::Asec().Apply(tensor);
}
