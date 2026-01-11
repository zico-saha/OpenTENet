#pragma once

#include <cmath>
#include <string>
#include <algorithm>
#include <numeric>
#include <numbers>
#include <stdexcept>

class Tensor;

namespace Math
{
	class BaseOperation
	{
	protected:
		static constexpr double EXP_BASE_LIMIT = 700.0;
		static constexpr double EPSILON_SCALE = 1e6;

	protected:
		Tensor Apply(const Tensor& tensor) const;

	public:
		virtual ~BaseOperation() = default;

		virtual double operator()(double x) const = 0;

		virtual Tensor f(const Tensor& tensor) const = 0;
	};
}
