#pragma once

#include "Utils.h"

#include <iostream>

class Complex
{
private:
	double real;
	double imaginary;

public:
	Complex(const double& _real, const double& _imaginary = 0.0);

	void Print() const;
};
