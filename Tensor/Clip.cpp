#include "Clip.h"
#include "Tensor.h"


Math::Clip::Clip(double min_value, double max_value)
{
	if (!std::isfinite(min_value))
	{
		throw std::invalid_argument("[Math] Clip Function failed: min_value must be a finite number.");
	}

	if (!std::isfinite(max_value))
	{
		throw std::invalid_argument("[Math] Clip Function failed: max_value must be a finite number.");
	}

	if (min_value > max_value)
	{
		throw std::invalid_argument("[Math] Clip Function failed: min_value cannot be greater than max_value.");
	}

	this->min_value = min_value;
	this->max_value = max_value;
}

double Math::Clip::operator()(double x) const
{
	return std::clamp(x, this->min_value, this->max_value);
}

Tensor Math::Clip::f(const Tensor& tensor) const
{
	return Math::Clip(this->min_value, this->max_value).Apply(tensor);
}
