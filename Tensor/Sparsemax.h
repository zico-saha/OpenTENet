#pragma once

#include "TensorActivation.h"

namespace Activation
{
	class Sparsemax : public TensorActivation
	{
	private:
		int axis;

		static constexpr double EPSILON = 1e-10;

	private:
		std::vector<double> __Sparsemax(const std::vector<double>& nums) const;

	public:
		explicit Sparsemax(int axis = -1);

		Tensor f(const Tensor& tensor) const override;

		Tensor df(const Tensor& tensor) const override;
	};
}
