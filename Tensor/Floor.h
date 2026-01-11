#pragma once

#include "BaseOperation.h"

namespace Math
{
	class Floor : public BaseOperation
	{
	public:
		double operator()(double x) const override;

		Tensor f(const Tensor& tensor) const override;
	};
}
