#include "Round.h"
#include "Tensor.h"


Math::Round::Round(int decimal_place)
{
	if (decimal_place < 0)
	{
		throw std::invalid_argument("[Math] Round Function failed: decimal_place cannot be negative.");
	}

	this->decimal_place = decimal_place;
	this->power_of_10 = std::pow(10.0, decimal_place);
}

double Math::Round::operator()(double x) const
{
	return std::round(x * this->power_of_10) / this->power_of_10;
}

Tensor Math::Round::f(const Tensor& tensor) const
{
	return Math::Round(this->decimal_place).Apply(tensor);
}
