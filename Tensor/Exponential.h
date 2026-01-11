#pragma once

#include "ScalarActivation.h"

namespace Activation
{
	class Exponential : public ScalarActivation
	{
	private:
		static constexpr double X_LIMIT = 700.0;

	public:
		using ScalarActivation::f;

		using ScalarActivation::df;

		double f(double x) const override;

		double df(double x) const override;
	};
}
