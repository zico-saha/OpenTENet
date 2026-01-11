#pragma once

#include "BaseOperation.h"

namespace Math
{
	class Power : public BaseOperation
	{
	private:
		double exponent;

	public:
		explicit Power(double exponent);

		double operator()(double x) const override;

		Tensor f(const Tensor& tensor) const override;
	};
}
