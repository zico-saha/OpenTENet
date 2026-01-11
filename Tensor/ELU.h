#pragma once

#include "ScalarActivation.h"

namespace Activation
{
	class ELU : public ScalarActivation
	{
	private:
		double alpha;

	public:
		explicit ELU(double alpha = 1.0);

		using ScalarActivation::f;

		using ScalarActivation::df;

		double f(double x) const override;

		double df(double x) const override;
	};
}
