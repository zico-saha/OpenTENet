#include "Acsc.h"
#include "Tensor.h"


double Math::Acsc::operator()(double x) const
{
	if (x > -1.0 && x < 1.0)
	{
		throw std::domain_error("[Math] Arc-cosecant Function failed: arccosecant is only defined for values in (-inf, -1] ∪ [1, inf).");
	}

	return std::asin(1.0 / x);
}

Tensor Math::Acsc::f(const Tensor& tensor) const
{
	return Math::Acsc().Apply(tensor);
}
