#include "Csc.h"
#include "Tensor.h"


double Math::Csc::operator()(double x) const
{
	double sin_value = std::sin(x);

	if (std::abs(sin_value) < std::numeric_limits<double>::epsilon() * Csc::EPSILON_SCALE)
	{
		throw std::domain_error("[Math] Cosecant Function failed: cosecant is undefined near multiples of `pi`.");
	}

	return 1.0 / sin_value;
}

Tensor Math::Csc::f(const Tensor& tensor) const
{
	return Math::Csc().Apply(tensor);
}
