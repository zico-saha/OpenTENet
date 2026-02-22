#pragma once

#include "Activation.h"
#include "LinAlg.h"
#include "TensorSlice.h"
#include "Utils.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <Windows.h>

#undef min
#undef max

class Math;
class TensorSlice;

class Tensor
{
	friend class TensorSlice;
	friend class Math;

private:
	int rank = 0;

	int volume = 0;

	std::shared_ptr<std::vector<double>> data;

	std::vector<int> shape;

	std::vector<int> strides;

	int start_point = 0;

	int end_point = 0;

	// ========== Constants ==========
	static constexpr double EPSILON_SCALE = 1e6;

public:
	using iterator = std::vector<double>::iterator;
	using const_iterator = std::vector<double>::const_iterator;

private:
	Tensor GetSlice(const int& _index) const;

	void SetSlice(const int& _index, const Tensor& _source);

	Tensor GetSliceChain(const std::vector<int>& _indices) const;

	void SetSliceChain(const std::vector<int>& _indices, const Tensor& _source);

	Tensor Apply(const std::function<double(double)>& _func) const;

public:
	Tensor() {}

	Tensor(const std::vector<int>& _shape, const double& _value = 0);

	Tensor(const std::vector<int>& _shape, const std::vector<double>& _data);

	Tensor(const LinAlg::Matrix& _matrix);

	Tensor(const Tensor& _tensor);

	iterator begin();

	iterator end();

	const_iterator begin() const;

	const_iterator end() const;

	void UniqueData();

	TensorSlice operator[](const int& _index);

	Tensor operator[](const int& _index) const;

	Tensor operator=(const Tensor& _tensor);

	Tensor operator+(const double& _value) const;

	Tensor operator-(const double& _value) const;

	Tensor operator*(const double& _value) const;

	Tensor operator/(const double& _value) const;

	Tensor operator+(const Tensor& _tensor) const;

	Tensor operator-(const Tensor& _tensor) const;

	Tensor operator*(const Tensor& _tensor) const;

	Tensor operator/(const Tensor& _tensor) const;

	void operator+=(const double& _value);

	void operator-=(const double& _value);

	void operator*=(const double& _value);

	void operator/=(const double& _value);

	void operator+=(const Tensor& _tensor);

	void operator-=(const Tensor& _tensor);

	void operator*=(const Tensor& _tensor);

	void operator/=(const Tensor& _tensor);

	Tensor Reshape(const std::vector<int>& _new_shape) const;

	Tensor ExpandRank(const int& _axis = 0) const;

	Tensor Flatten(const int& _axis_from, const int& _axis_upto) const;

	Tensor Slice(const int& _axis, const int& _index) const;

	Tensor Slice(const int& _axis, const int& _index_from, const int& _index_upto) const;

	void Append(const Tensor& _tensor, const int& _axis = -1);

	void Insert(const Tensor& _tensor, const int& _axis = -1, const int& _index = 0);

	static Tensor Concat(const std::vector<Tensor>& _tensors, const int& _axis = -1);

	static Tensor Stack(const std::vector<Tensor>& _tensors, const int& _axis = 0);

	Tensor Pad(const int& _axis, const int& _pad_before_size, const int& _pad_after_size, const double& _value = 0.0) const;

	Tensor Tile(const std::vector<int>& _repetitions) const;

	Tensor Broadcast(const std::vector<int>& _shape) const;

	Tensor Transpose(const std::vector<int>& _permutation) const;

	static Tensor MatMul(const Tensor& _tensor_1, const Tensor& _tensor_2);

	Tensor MatMul(const Tensor& _tensor) const;

	static Tensor TensorDot(const Tensor& _tensor_1, const Tensor& _tensor_2, const std::vector<int>& _contract_axes_1, const std::vector<int>& _contract_axes_2);

	Tensor Convolve(const Tensor& _filter, const std::vector<int>& _strides, const std::vector<int>& _padding);

	Tensor MaxPool(const std::vector<int>& _pool_shape, const std::vector<int>& _strides = {});

	Tensor MinPool(const std::vector<int>& _pool_shape, const std::vector<int>& _strides = {});

	Tensor AvgPool(const std::vector<int>& _pool_shape, const std::vector<int>& _strides = {});

	Tensor Sign(const bool& _heaviside = false) const;

	Tensor ReduceSum(const int& _axis = 0) const;

	Tensor ReduceMean(const int& _axis = 0) const;

	Tensor ReduceVar(const int& _axis = 0, const bool& _inference = false) const;

	Tensor ReduceMax(const int& _axis = 0) const;

	Tensor ReduceMin(const int& _axis = 0) const;

	double Sum() const;

	double Mean() const;

	double Var(const bool& _inference = false) const;

	double Max() const;

	double Min() const;

	Tensor Activate(const Activation::BaseActivation& _activation_func) const;

	Tensor ActivateDerivative(const Activation::BaseActivation& _activation_func) const;

	int Rank() const;

	int Volume() const;

	std::vector<int> Shape() const;

	bool IsEmpty() const;

	bool IsScalar() const;

	void Clear();

	void Print(const int& _depth = 0) const;

	double ToScalar() const;

	std::vector<double> ToVector() const;

	std::vector<std::vector<double>> ToMatrix() const;
};
