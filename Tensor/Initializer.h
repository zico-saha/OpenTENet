#pragma once

#include "Tensor.h"
#include "Utils.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

class Initializer
{
private:
	std::vector<int> shape;

	int rank;

	int volume;

	std::optional<unsigned int> seed;

	mutable std::mt19937 generator;

private:
	void InitializeGenerator();

public:
	Initializer();

	Initializer(const std::vector<int>& _shape, const std::optional<unsigned int>& _seed = std::nullopt);

	Tensor Zeros() const;

	Tensor Ones() const;

	Tensor Constant(const double& _value) const;

	Tensor Identity(const std::pair<int, int>& _matrix_axes, const double& scale = 1.0) const;

	Tensor RandomNormal(const double& _mean = 0.0, const double& _std_dev = 1.0) const;

	Tensor RandomUniform(const double& _min_val = 0.0, const double& _max_val = 1.0) const;

	Tensor TruncatedNormal(const double& _mean = 0.0, const double& _std_dev = 1.0, const double& _truncate_std_dev_scale = 2.0) const;

	Tensor GlorotNormal(const int& _fan_in, const int& _fan_out) const;

	Tensor GlorotUniform(const int& _fan_in, const int& _fan_out) const;

	Tensor HeNormal(const int& _fan_in) const;

	Tensor HeUniform(const int& _fan_in) const;

	Tensor LecunNormal(const int& _fan_in) const;

	Tensor LecunUniform(const int& _fan_in) const;

	Tensor Orthogonal(const double& gain = 1.0) const;
};
