#include "Sqrt.h"
#include "Tensor.h"


double Math::Sqrt::operator()(double x) const
{
	if (x < 0.0)
	{
		throw std::domain_error("[Math] Sqrt Function failed: negative value found in input Tensor, " + std::to_string(x));
	}

	return std::sqrt(x);
}

Tensor Math::Sqrt::f(const Tensor& tensor) const
{
	return Math::Sqrt().Apply(tensor);
}
