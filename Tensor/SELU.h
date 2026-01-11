#pragma once

#include "ScalarActivation.h"

namespace Activation
{
    class SELU : public ScalarActivation
    {
    private:
        static constexpr double LAMBDA = 1.0507009873554804934193349852946;
        static constexpr double ALPHA = 1.6732632423543772848170429916717;

    public:
        using ScalarActivation::f;

        using ScalarActivation::df;

        double f(double x) const override;

        double df(double x) const override;
    };
};
