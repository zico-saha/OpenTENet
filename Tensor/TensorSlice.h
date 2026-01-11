#pragma once

#include "Activation.h"
#include "Math.h"

#include <vector>
#include <string>
#include <numbers>
#include <iterator>
#include <memory>
#include <functional>

class Tensor;

class TensorSlice
{
private:
	Tensor *root_parent;

	std::vector<int> index_chain;

	struct SliceInfo
	{
		std::shared_ptr<std::vector<double>> data;
		int start_offset = 0;
		int end_offset = 0;
		std::vector<int> shape;
		std::vector<int> strides;
	};

private:
	TensorSlice(Tensor *_ptr, const std::vector<int> &_chain)
		: root_parent(_ptr), index_chain(_chain)
	{
	}

	Tensor _Tensor() const;

	SliceInfo _GetDirectAccess() const;

public:
	using iterator = std::vector<double>::iterator;
	using const_iterator = std::vector<double>::const_iterator;

public:
	TensorSlice(Tensor *_ptr, const int& _index)
		: root_parent(_ptr), index_chain({_index})
	{
	}

	operator Tensor() const;

	TensorSlice& operator=(const Tensor& _tensor);

	TensorSlice operator[](const int& _index);

	Tensor operator[](const int& _index) const;

	iterator begin();

	iterator end();

	const_iterator begin() const;

	const_iterator end() const;

	Tensor operator+(const double& _value) const;

	Tensor operator-(const double& _value) const;

	Tensor operator*(const double& _value) const;

	Tensor operator/(const double& _value) const;

	Tensor operator+(const Tensor& _tensor) const;

	Tensor operator-(const Tensor& _tensor) const;

	Tensor operator*(const Tensor& _tensor) const;

	Tensor operator/(const Tensor& _tensor) const;

	TensorSlice &operator+=(const double& _value);

	TensorSlice &operator-=(const double& _value);

	TensorSlice &operator*=(const double& _value);

	TensorSlice &operator/=(const double& _value);

	TensorSlice& operator+=(const Tensor& _tensor);

	TensorSlice& operator-=(const Tensor& _tensor);

	TensorSlice& operator*=(const Tensor& _tensor);

	TensorSlice& operator/=(const Tensor& _tensor);

	Tensor Reshape(const std::vector<int>& _new_shape) const;

	Tensor ExpandRank(const int& _axis = 0) const;

	Tensor Flatten(const int& _axis_from, const int& _axis_upto) const;

	Tensor Slice(const int& _axis, const int& _index) const;

	Tensor Slice(const int& _axis, const int& _index_from, const int& _index_upto) const;

	Tensor Pad(const int& _axis, const int& _pad_before_size, const int& _pad_after_size, const double& _value = 0.0) const;

	Tensor Tile(const std::vector<int>& _repetitions) const;

	Tensor Broadcast(const std::vector<int>& _shape) const;

	Tensor Transpose(const std::vector<int>& _permutation) const;

	Tensor MatMul(const Tensor& _tensor) const;

	Tensor Convolve(const Tensor& _filter, const std::vector<int>& _strides, const std::vector<int>& _padding);

	Tensor MaxPool(const std::vector<int>& _pool_shape, const std::vector<int>& _strides = {});

	Tensor MinPool(const std::vector<int>& _pool_shape, const std::vector<int>& _strides = {});

	Tensor AvgPool(const std::vector<int>& _pool_shape, const std::vector<int>& _strides = {});

	Tensor Sign(const bool& _heaviside = false) const;

	Tensor ReduceSum(const int& _axis) const;

	Tensor ReduceMean(const int& _axis) const;

	Tensor ReduceVar(const int& _axis, const bool& _inference = false) const;

	Tensor ReduceMax(const int& _axis) const;

	Tensor ReduceMin(const int& _axis) const;

	double Sum() const;

	double Mean() const;

	double Var(const bool& _inference = false) const;

	double Max() const;

	double Min() const;

	Tensor MathOps(const Math::BaseOperation& _math_func) const;

	Tensor Activate(const Activation::BaseActivation& _activation_func) const;

	Tensor ActivateDerivative(const Activation::BaseActivation& _activation_func) const;

	int Rank() const;

	int Volume() const;

	std::vector<int> Shape() const;

	bool IsEmpty() const;

	bool IsScalar() const;

	void Print(const int& _depth = 0) const;

	double ToScalar() const;

	std::vector<double> ToVector() const;

	std::vector<std::vector<double>> ToMatrix() const;
};
