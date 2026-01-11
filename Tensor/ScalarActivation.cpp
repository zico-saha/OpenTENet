#include "ScalarActivation.h"
#include "Tensor.h"


bool Activation::ScalarActivation::isScalar() const
{
	return true;
}

Tensor Activation::ScalarActivation::f(const Tensor& tensor) const
{
	std::vector<double> activated_data;
	activated_data.reserve(tensor.Volume());

	for (const double& value : tensor)
	{
		activated_data.push_back(this->f(value));
	}

	return Tensor(tensor.Shape(), activated_data);
}

Tensor Activation::ScalarActivation::df(const Tensor& tensor) const
{
	std::vector<double> activated_data;
	activated_data.reserve(tensor.Volume());

	for (const double& value : tensor)
	{
		activated_data.push_back(this->df(value));
	}

	return Tensor(tensor.Shape(), activated_data);
}
