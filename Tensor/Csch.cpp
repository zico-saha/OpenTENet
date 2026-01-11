#include "Csch.h"
#include "Tensor.h"


double Math::Csch::operator()(double x) const
{
	double sinh_value = std::sinh(x);

	if (std::abs(sinh_value) < std::numeric_limits<double>::epsilon() * Csch::EPSILON_SCALE)
	{
		throw std::domain_error("[Math] Hyperbolic Cosecant Function failed: hyperbolic cosecant is undefined at zero.");
	}

	return 1.0 / sinh_value;
}

Tensor Math::Csch::f(const Tensor& tensor) const
{
	return Math::Csch().Apply(tensor);
}
