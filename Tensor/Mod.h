#pragma once

#include "BaseOperation.h"

namespace Math
{
	class Mod : public BaseOperation
	{
	private:
		double mod_value;

	public:
		explicit Mod(double mod_value);

		double operator()(double x) const override;

		Tensor f(const Tensor& tensor) const override;
	};
}
