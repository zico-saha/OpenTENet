#pragma once

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <numbers>
#include <stdexcept>

class Tensor;

namespace Activation
{
    class BaseActivation
    {
    public:
        virtual ~BaseActivation() = default;

        virtual bool isScalar() const = 0;

        virtual double f(double x) const = 0;

        virtual double df(double x) const = 0;

        virtual Tensor f(const Tensor& tensor) const = 0;

        virtual Tensor df(const Tensor& tensor) const = 0;
    };
}
