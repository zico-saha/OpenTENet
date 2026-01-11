#pragma once

#include "ScalarActivation.h"

namespace Activation
{
	class SoftShrink : public ScalarActivation
	{
	private:
		double threshold;

	public:
		explicit SoftShrink(double threshold = 0.5);

		using ScalarActivation::f;

		using ScalarActivation::df;

		double f(double x) const override;

		double df(double x) const override;
	};
}
