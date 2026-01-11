#include "Acot.h"
#include "Tensor.h"


double Math::Acot::operator()(double x) const
{
	double acot_value;

	if (std::abs(x) < std::numeric_limits<double>::epsilon() * Acot::EPSILON_SCALE)
	{
		acot_value = std::numbers::pi / 2.0;
	}
	else
	{
		acot_value = std::atan(1.0 / x);
		acot_value += (x < 0.0) ? std::numbers::pi : 0.0;
	}

	return acot_value;
}

Tensor Math::Acot::f(const Tensor& tensor) const
{
	return Math::Acot().Apply(tensor);
}
