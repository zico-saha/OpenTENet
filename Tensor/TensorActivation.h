#pragma once

#include "BaseActivation.h"

namespace Activation
{
    class TensorActivation : public BaseActivation
    {
    public:
        bool isScalar() const override;

        double f(double x) const override;

        double df(double x) const override;

        virtual Tensor f(const Tensor& tensor) const = 0;

        virtual Tensor df(const Tensor& tensor) const = 0;
    };
}
