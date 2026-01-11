#pragma once

#include "BaseOperation.h"

namespace Math
{
	class Clip : public BaseOperation
	{
	private:
		double min_value;
		double max_value;

	public:
		explicit Clip(double min_value, double max_value);

		double operator()(double x) const override;

		Tensor f(const Tensor& tensor) const override;
	};
}
