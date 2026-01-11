#include "Asin.h"
#include "Tensor.h"


double Math::Asin::operator()(double x) const
{
	if (x < -1.0 || x > 1.0)
	{
		throw std::domain_error("[Math] Arc-sine Function failed: arcsine is only defined for values in [-1, 1].");
	}

	return std::asin(x);
}

Tensor Math::Asin::f(const Tensor& tensor) const
{
	return Math::Asin().Apply(tensor);
}
