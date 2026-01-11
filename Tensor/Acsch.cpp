#include "Acsch.h"
#include "Tensor.h"


double Math::Acsch::operator()(double x) const
{
	if (std::abs(x) < std::numeric_limits<double>::epsilon() * Acsch::EPSILON_SCALE)
	{
		throw std::domain_error("[Math] Inverse Hyperbolic Cosecant Function failed: inverse hyperbolic cosecant is undefined at zero.");
	}

	return std::asinh(1.0 / x);
}

Tensor Math::Acsch::f(const Tensor& tensor) const
{
	return Math::Acsch().Apply(tensor);
}
