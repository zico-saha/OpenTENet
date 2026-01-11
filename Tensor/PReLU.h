#pragma once

#include "ScalarActivation.h"

namespace Activation
{
	class PReLU : public ScalarActivation
	{
    private:
        double alpha;

    public:
        explicit PReLU(double alpha = 0.01);

        using ScalarActivation::f;

        using ScalarActivation::df;

        double f(double x) const override;

        double df(double x) const override;
	};
}
