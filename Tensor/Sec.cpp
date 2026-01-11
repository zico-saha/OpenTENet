#include "Sec.h"
#include "Tensor.h"


double Math::Sec::operator()(double x) const
{
	double cos_value = std::cos(x);

	if (std::abs(cos_value) < std::numeric_limits<double>::epsilon() * Sec::EPSILON_SCALE)
	{
		throw std::domain_error("[Math] Secant Function failed: secant is undefined near odd multiples of `pi`/2.");
	}

	return 1.0 / cos_value;
}

Tensor Math::Sec::f(const Tensor& tensor) const
{
	return Math::Sec().Apply(tensor);
}
