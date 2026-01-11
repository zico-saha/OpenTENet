#pragma once

#include "TensorActivation.h"

namespace Activation
{
	class Softmax : public TensorActivation
	{
	private:
		int axis;

	public:
		explicit Softmax(int axis = -1);

		Tensor f(const Tensor& tensor) const override;

		Tensor df(const Tensor& tensor) const override;
	};
};
