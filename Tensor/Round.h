#pragma once

#include "BaseOperation.h"

namespace Math
{
	class Round : public BaseOperation
	{
	private:
		int decimal_place;
		double power_of_10;

	public:
		explicit Round(int decimal_place = 2);

		double operator()(double x) const override;

		Tensor f(const Tensor& tensor) const override;
	};
}
