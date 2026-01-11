#pragma once

#include "ScalarActivation.h"

namespace Activation
{
    class LeakyReLU : public ScalarActivation
    {
    private:
        double alpha;

    public:
        explicit LeakyReLU(double alpha = 0.01);

        using ScalarActivation::f;

        using ScalarActivation::df;

        double f(double x) const override;

        double df(double x) const override;
    };
};
