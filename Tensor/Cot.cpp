#include "Cot.h"
#include "Tensor.h"


double Math::Cot::operator()(double x) const
{
	double sin_value = std::sin(x);

	if (std::abs(sin_value) < std::numeric_limits<double>::epsilon() * Cot::EPSILON_SCALE)
	{
		throw std::domain_error("[Math] Cotangent Function failed: cotangent is undefined near multiples of `pi`.");
	}

	return 1.0 / std::tan(x);
}

Tensor Math::Cot::f(const Tensor& tensor) const
{
	return Math::Cot().Apply(tensor);
}
