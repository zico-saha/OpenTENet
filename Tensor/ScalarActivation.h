#pragma once

#include "BaseActivation.h"

namespace Activation
{
	class ScalarActivation : public BaseActivation
	{
	public:
		bool isScalar() const override;

		virtual double f(double x) const = 0;

		virtual double df(double x) const = 0;

		Tensor f(const Tensor& tensor) const override;

		Tensor df(const Tensor& tensor) const override;
	};
};
