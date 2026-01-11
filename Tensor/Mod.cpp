#include "Mod.h"
#include "Tensor.h"


Math::Mod::Mod(double mod_value)
{
	if (std::abs(mod_value) < std::numeric_limits<double>::epsilon() * Mod::EPSILON_SCALE)
	{
		throw std::domain_error("[Math] Modulus Function failed: modulus value cannot be 0 or (~0).");
	}

	this->mod_value = mod_value;
}

double Math::Mod::operator()(double x) const
{
	return std::fmod(x, this->mod_value);
}

Tensor Math::Mod::f(const Tensor& tensor) const
{
	return Math::Mod(this->mod_value).Apply(tensor);
}
