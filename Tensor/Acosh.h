#pragma once

#include "BaseOperation.h"

namespace Math
{
	class Acosh : public BaseOperation
	{
	public:
		double operator()(double x) const override;

		Tensor f(const Tensor& tensor) const override;
	};
}
