#include "Log.h"
#include "Tensor.h"


Math::Log::Log(double base)
{
	if (base <= 0.0 || base < std::numeric_limits<double>::epsilon() * Log::EPSILON_SCALE)
	{
		throw std::domain_error("[Math] Logarithm Function failed: base cannot be zero or negative.");
	}

	if (std::abs(base - 1.0) < std::numeric_limits<double>::epsilon() * Log::EPSILON_SCALE)
	{
		throw std::domain_error("[Math] Logarithm Function failed: base cannot be 1.");
	}

	this->base = base;
	this->log_base = std::log(base);
}

double Math::Log::operator()(double x) const
{
	if (x <= 0.0 || x < std::numeric_limits<double>::epsilon() * Math::Log::EPSILON_SCALE)
	{
		throw std::invalid_argument("[Math] Logarithm Function failed: detected non-positive value, " + std::to_string(x) + " - logarithm is undefined.");
	}

	return std::log(x) / this->log_base;
}

Tensor Math::Log::f(const Tensor& tensor) const
{
	return Math::Log(this->base).Apply(tensor);
}
