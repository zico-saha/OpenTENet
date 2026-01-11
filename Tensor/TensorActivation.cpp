#include "TensorActivation.h"


bool Activation::TensorActivation::isScalar() const
{
    return false;
}

double Activation::TensorActivation::f(double x) const
{
    throw std::logic_error("[Activation] Tensor Activation failed: Scalar activation not supported for TensorActivation.");
}

double Activation::TensorActivation::df(double x) const
{
    throw std::logic_error("[Activation] Tensor Activation failed: Scalar derivative not supported for TensorActivation.");
}
