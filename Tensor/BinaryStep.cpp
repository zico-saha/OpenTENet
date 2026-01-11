#include "BinaryStep.h"


double Activation::BinaryStep::f(double x) const
{
    return (x > 0) ? 1.0 : 0.0;
}

double Activation::BinaryStep::df(double x) const
{
    return 0.0;
}
