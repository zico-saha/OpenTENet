#pragma once

#include "BaseOperation.h"

namespace Math
{
	class Log : public BaseOperation
	{
	private:
		double base;
		double log_base;

	public:
		explicit Log(double base = std::numbers::e);

		double operator()(double x) const override;

		Tensor f(const Tensor& tensor) const override;
	};
}
