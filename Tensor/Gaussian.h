#pragma once

#include "ScalarActivation.h"

namespace Activation
{
	class Gaussian : public ScalarActivation
	{
	private:
		double center;
		double std_dev;
		double scale;

	public:
		explicit Gaussian(double center = 0.0, double std_dev = 1.0, double scale = 1.0);

		using ScalarActivation::f;

		using ScalarActivation::df;

		double f(double x) const override;

		double df(double x) const override;
	};
}
