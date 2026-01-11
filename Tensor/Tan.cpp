#include "Tan.h"
#include "Tensor.h"


double Math::Tan::operator()(double x) const
{
	double cos_value = std::cos(x);

	if (std::abs(cos_value) < std::numeric_limits<double>::epsilon() * Tan::EPSILON_SCALE)
	{
		throw std::domain_error("[Math] Tangent Function failed: tangent is undefined near odd multiples of `pi`/2.");
	}

	return std::tan(x);
}

Tensor Math::Tan::f(const Tensor& tensor) const
{
	return Math::Tan().Apply(tensor);
}
