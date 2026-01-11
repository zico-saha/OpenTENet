#pragma once

#include "ScalarActivation.h"

namespace Activation
{
	class SquarePlus : public ScalarActivation
	{
	private:
		double smoothness;

	public:
		explicit SquarePlus(double smoothness = 4.0);

		using ScalarActivation::f;

		using ScalarActivation::df;

		double f(double x) const override;

		double df(double x) const override;
	};
}
