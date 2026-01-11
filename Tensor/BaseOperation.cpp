#include "BaseOperation.h"
#include "Tensor.h"


Tensor Math::BaseOperation::Apply(const Tensor& tensor) const
{
	if (tensor.IsEmpty())
	{
		throw std::runtime_error("[Math] Math Operation failed: cannot perform mathematical operation on empty Tensor.");
	}

	std::vector<double> result_data;
	result_data.reserve(tensor.Volume());

	for (const double& value : tensor)
	{
		result_data.push_back((*this)(value));
	}

	return Tensor(tensor.Shape(), result_data);
}
