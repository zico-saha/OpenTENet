#include "Coth.h"
#include "Tensor.h"


double Math::Coth::operator()(double x) const
{
	double sinh_value = std::sinh(x);

	if (std::abs(sinh_value) < std::numeric_limits<double>::epsilon() * Coth::EPSILON_SCALE)
	{
		throw std::domain_error("[Math] Hyperbolic Cotangent Function failed: hyperbolic cotangent is undefined at zero.");
	}

	return std::cosh(x) / sinh_value;
}

Tensor Math::Coth::f(const Tensor& tensor) const
{
	return Math::Coth().Apply(tensor);
}
