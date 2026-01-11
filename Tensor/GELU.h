#pragma once

#include "ScalarActivation.h"

namespace Activation
{
	class GELU : public ScalarActivation
	{
	private:
		bool approx;
		static constexpr double SQRT_2_OVER_PI = 0.7978845608028654;
		static constexpr double COEFF = 0.044715;

	public:
		explicit GELU(bool approx = true);

		using ScalarActivation::f;

		using ScalarActivation::df;

		double f(double x) const override;

		double df(double x) const override;
	};
}
