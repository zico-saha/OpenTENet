#include "Exp.h"
#include "Tensor.h"


double Math::Exp::operator()(double x) const
{
	if (x > Math::Exp::EXP_BASE_LIMIT)
	{
		throw std::invalid_argument("[Math] Exponent Function failed: detected large value, " + std::to_string(x) + " - may cause overflow.");
	}

	return std::exp(x);
}

Tensor Math::Exp::f(const Tensor& tensor) const
{
	return Math::Exp().Apply(tensor);
}
