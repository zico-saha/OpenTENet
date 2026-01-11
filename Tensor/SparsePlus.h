#pragma once

#include "ScalarActivation.h"

namespace Activation
{
    class SparsePlus : public ScalarActivation
    {
    public:
        using ScalarActivation::f;

        using ScalarActivation::df;

        double f(double x) const override;

        double df(double x) const override;
    };
};
