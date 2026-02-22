#include "Complex.h"

Complex::Complex(const double& _real, const double& _imaginary)
{
	if (!Utils::IsValidData<double>({ _real }))
	{
		throw std::invalid_argument("[Complex] Constructor failed: invalid real-value.");
	}

	if (!Utils::IsValidData<double>({ _imaginary }))
	{
		throw std::invalid_argument("[Complex] Constructor failed: invalid imaginary-value.");
	}

	this->real = _real;
	this->imaginary = _imaginary;
}

void Complex::Print() const
{
	std::cout << "<" << this->real << "," << this->imaginary << ">";
}
