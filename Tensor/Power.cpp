#include "Power.h"
#include "Tensor.h"


Math::Power::Power(double exponent)
{
	this->exponent = exponent;
}

double Math::Power::operator()(double x) const
{
	if (x < 0.0 && std::fmod(exponent, 1.0) != 0.0)
	{
		throw std::domain_error("[Math] Power Function failed: negative base detected with non-integer exponent -results non-real number.");
	}

	return std::pow(x, this->exponent);
}

Tensor Math::Power::f(const Tensor& tensor) const
{
	return Math::Power(this->exponent).Apply(tensor);
}
