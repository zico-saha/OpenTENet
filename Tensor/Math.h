#pragma once

#include <cmath>
#include <string>
#include <algorithm>
#include <numeric>
#include <numbers>
#include <stdexcept>
#include <vector>

class Tensor;

class Math
{
private:
	static constexpr double EXP_BASE_LIMIT = 700.0;
	static constexpr double EPSILON_SCALE = 1e6;

public:
	static Tensor Abs(const Tensor& _tensor);

	static Tensor Ceil(const Tensor& _tensor);

	static Tensor Clip(const Tensor& _tensor, const double& _min_value, const double& _max_value);

	static Tensor Exp(const Tensor& _tensor);

	static Tensor Floor(const Tensor& _tensor);

	static Tensor Log(const Tensor& _tensor, const double& _base = std::numbers::e);

	static Tensor Mod(const Tensor& _tensor, const double& _mod_value);

	static Tensor Power(const Tensor& _tensor, const double& _exponent);

	static Tensor Round(const Tensor& _tensor, const int& _decimal_place = 2);

	static Tensor Sqrt(const Tensor& _tensor);

	static Tensor Sin(const Tensor& _tensor);

	static Tensor Cos(const Tensor& _tensor);

	static Tensor Tan(const Tensor& _tensor);

	static Tensor Csc(const Tensor& _tensor);

	static Tensor Sec(const Tensor& _tensor);

	static Tensor Cot(const Tensor& _tensor);

	static Tensor Asin(const Tensor& _tensor);

	static Tensor Acos(const Tensor& _tensor);

	static Tensor Atan(const Tensor& _tensor);

	static Tensor Acsc(const Tensor& _tensor);

	static Tensor Asec(const Tensor& _tensor);

	static Tensor Acot(const Tensor& _tensor);

	static Tensor Sinh(const Tensor& _tensor);

	static Tensor Cosh(const Tensor& _tensor);

	static Tensor Tanh(const Tensor& _tensor);

	static Tensor Csch(const Tensor& _tensor);

	static Tensor Sech(const Tensor& _tensor);

	static Tensor Coth(const Tensor& _tensor);

	static Tensor Asinh(const Tensor& _tensor);

	static Tensor Acosh(const Tensor& _tensor);

	static Tensor Atanh(const Tensor& _tensor);

	static Tensor Acsch(const Tensor& _tensor);

	static Tensor Asech(const Tensor& _tensor);

	static Tensor Acoth(const Tensor& _tensor);
};
