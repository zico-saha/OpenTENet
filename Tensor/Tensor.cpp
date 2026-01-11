#include "Tensor.h"

// ========================================
// [Private] TensorSlice Helper Methods
// ========================================
Tensor Tensor::GetSlice(const int& _index) const
{
	if (this->shape.empty())
	{
		throw std::runtime_error("[Tensor] GetSlice failed: cannot index a rank-0 or empty Tensor");
	}

	if (_index < 0)
	{
		throw std::out_of_range("[Tensor] GetSlice failed: index must be >= 0.");
	}

	if (_index >= this->shape[0])
	{
		throw std::out_of_range("[Tensor] GetSlice failed: index must be < " + std::to_string(this->shape[0]) + ".");
	}

	int slice_volume = 1;
	for (int i = 1; i < this->rank; i++)
	{
		slice_volume *= this->shape[i];
	}

	std::vector<double> slice_data;
	slice_data.reserve(slice_volume);

	int start_pos = this->start_point + (_index * this->strides[0]);
	int end_pos = start_pos + slice_volume;

	auto start_ptr = this->data->begin() + start_pos;
	auto end_ptr = this->data->begin() + end_pos;

	slice_data.insert(slice_data.end(), start_ptr, end_ptr);

	std::vector<int> slice_shape(this->shape.begin() + 1, this->shape.end());

	return Tensor(slice_shape, slice_data);
}

void Tensor::SetSlice(const int& _index, const Tensor& _source)
{
	if (this->shape.empty())
	{
		throw std::runtime_error("[Tensor] SetSlice failed: cannot index a rank-0 or empty Tensor");
	}

	if (_index < 0)
	{
		throw std::out_of_range("[Tensor] SetSlice failed: index must be >= 0.");
	}

	if (_index >= this->shape[0])
	{
		throw std::out_of_range("[Tensor] SetSlice failed: index must be < " + std::to_string(this->shape[0]) + ".");
	}

	int slice_volume = 1;
	for (int i = 1; i < this->rank; i++)
	{
		slice_volume *= this->shape[i];
	}

	if (_source.volume != slice_volume)
	{
		throw std::invalid_argument("[Tensor] SetSlice failed: source volume must match slice volume.");
	}

	this->UniqueData();

	int start_pos = this->start_point + (_index * this->strides[0]);

	auto src_start = _source.data->begin() + _source.start_point;
	auto src_end = _source.data->begin() + _source.end_point;
	auto dest = this->data->begin() + start_pos;

	std::copy(src_start, src_end, dest);
}

Tensor Tensor::GetSliceChain(const std::vector<int>& _indices) const
{
	Tensor result = *this;

	for (int idx : _indices)
	{
		result = result.GetSlice(idx);
	}

	return result;
}

void Tensor::SetSliceChain(const std::vector<int>& _indices, const Tensor& _source)
{
	if (_indices.empty())
	{
		throw std::invalid_argument("[Tensor] SetSliceChain failed: index chain cannot be empty.");
	}

	if (_indices.size() == 1)
	{
		this->SetSlice(_indices[0], _source);
		return;
	}

	this->UniqueData();

	Tensor temp = this->GetSlice(_indices[0]);

	std::vector<int> remaining(_indices.begin() + 1, _indices.end());

	temp.SetSliceChain(remaining, _source);

	this->SetSlice(_indices[0], temp);
}

// ========================================
// Tensor Constructors
// ========================================
Tensor::Tensor(const std::vector<int>& _shape, const double& _value)
{
	if (!Utils::IsValidData<double>({_value}))
	{
		throw std::invalid_argument("[Tensor] Constructor failed: invalid value.");
	}

	if (shape.empty())
	{
		this->rank = 0;
		this->volume = 1;

		this->data = std::make_shared<std::vector<double>>(1, _value);

		this->start_point = 0;
		this->end_point = this->volume;

		return;
	}

	if (!Utils::IsAllPositive(_shape))
	{
		throw std::invalid_argument("[Tensor] Constructor failed: all shape dimensions must be > 0.");
	}

	if (Utils::IsVolumeOverflow(_shape))
	{
		throw std::overflow_error("[Tensor] Constructor failed: shape too large, potential overflow.");
	}

	this->rank = _shape.size();
	this->volume = Utils::ShapeToVolume(_shape);

	this->shape = _shape;
	this->strides = Utils::ShapeToStrides(_shape);

	this->data = std::make_shared<std::vector<double>>(this->volume, _value);

	this->start_point = 0;
	this->end_point = this->volume;
}

Tensor::Tensor(const std::vector<int>& _shape, const std::vector<double>& _data)
{
	if (_data.empty())
	{
		throw std::invalid_argument("[Tensor] Constructor failed: empty data.");
	}

	if (!Utils::IsValidData(_data))
	{
		throw std::invalid_argument("[Tensor] Constructor failed: invalid value.");
	}

	if (_shape.empty())
	{
		if (_data.size() > 1)
		{
			throw std::invalid_argument("[Tensor] Constructor failed: single value expected for rank-0 tensor.");
		}

		this->rank = 0;
		this->volume = static_cast<int>(_data.size());

		this->data = std::make_shared<std::vector<double>>(_data);

		this->start_point = 0;
		this->end_point = this->volume;

		return;
	}

	if (!Utils::IsAllPositive(_shape))
	{
		throw std::invalid_argument("[Tensor] Constructor failed: all shape dimensions must be > 0.");
	}

	if (Utils::IsVolumeOverflow(_shape))
	{
		throw std::overflow_error("[Tensor] Constructor failed: shape too large, potential overflow.");
	}

	this->rank = _shape.size();
	this->volume = Utils::ShapeToVolume(_shape);

	this->shape = _shape;
	this->strides = Utils::ShapeToStrides(_shape);

	if (this->volume != static_cast<int>(_data.size()))
	{
		throw std::invalid_argument("[Tensor] Constructor failed: shape-volume mismatch with data-volume");
	}

	this->data = std::make_shared<std::vector<double>>(_data);

	this->start_point = 0;
	this->end_point = this->volume;
}

Tensor::Tensor(const Tensor& _tensor)
{
	this->rank = _tensor.rank;
	this->volume = _tensor.volume;

	this->shape = _tensor.shape;
	this->strides = _tensor.strides;

	this->start_point = 0;
	this->end_point = _tensor.volume;

	auto start_ptr = (*_tensor.data).begin() + _tensor.start_point;
	auto end_ptr = (*_tensor.data).begin() + _tensor.end_point;

	this->data = std::make_shared<std::vector<double>>(start_ptr, end_ptr);
}

// ========================================
// Tensor Iterator(s)
// ========================================
Tensor::iterator Tensor::begin()
{
	return this->data->begin() + this->start_point;
}

Tensor::iterator Tensor::end()
{
	return this->data->begin() + this->end_point;
}

Tensor::const_iterator Tensor::begin() const
{
	return this->data->begin() + this->start_point;
}

Tensor::const_iterator Tensor::end() const
{
	return this->data->begin() + this->end_point;
}

// ========================================
// Tensor Unique Memory Allocation Method
// ========================================
void Tensor::UniqueData()
{
	if (!this->IsEmpty() && this->data.use_count() > 1)
	{
		auto start_ptr = this->data->begin() + this->start_point;
		auto end_ptr = this->data->begin() + this->end_point;

		this->data = std::make_shared<std::vector<double>>(start_ptr, end_ptr);

		this->start_point = 0;
		this->end_point = this->volume;
	}
}

// ========================================
// Tensor Indexing Operator
// ========================================
TensorSlice Tensor::operator[](const int& _index)
{
	if (this->shape.empty())
	{
		throw std::runtime_error("[Tensor] Indexing failed: cannot index a rank-0 or empty Tensor");
	}

	if (_index < 0)
	{
		throw std::out_of_range("[Tensor] Indexing failed: index must be >= 0.");
	}

	if (_index >= this->shape[0])
	{
		throw std::out_of_range("[Tensor] Indexing failed: index must be < " + std::to_string(this->shape[0]) + ".");
	}

	return TensorSlice(this, _index);
}

Tensor Tensor::operator[](const int& _index) const
{
	if (this->shape.empty())
	{
		throw std::runtime_error("[Tensor] Indexing failed: cannot index a rank-0 or empty Tensor");
	}

	if (_index < 0)
	{
		throw std::out_of_range("[Tensor] Indexing failed: index must be >= 0.");
	}

	if (_index >= this->shape[0])
	{
		throw std::out_of_range("[Tensor] Indexing failed: index must be < " + std::to_string(this->shape[0]) + ".");
	}

	int slice_volume = 1;
	for (int i = 1; i < this->rank; i++)
	{
		slice_volume *= shape[i];
	}

	return this->GetSlice(_index);
}

// ========================================
// Tensor Assignment Operator (Deep Copy)
// ========================================
Tensor Tensor::operator=(const Tensor& _tensor)
{
	if (this == &_tensor)
	{
		return *this;
	}

	this->rank = _tensor.rank;
	this->volume = _tensor.volume;

	this->shape = _tensor.shape;
	this->strides = _tensor.strides;

	auto start_ptr = _tensor.data->begin() + _tensor.start_point;
	auto end_ptr = _tensor.data->begin() + _tensor.end_point;
	this->data = std::make_shared<std::vector<double>>(start_ptr, end_ptr);

	this->start_point = 0;
	this->end_point = _tensor.volume;

	return *this;
}

// ========================================
// Tensor Arithmetic Operator(s)
// ========================================
Tensor Tensor::operator+(const double& _value) const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Tensor] Addition failed: cannot perform addition on empty Tensor.");
	}

	if (!Utils::IsValidData<double>({ _value }))
	{
		throw std::invalid_argument("[Tensor] Addition failed: invalid value.");
	}

	std::vector<double> result_data(this->volume, 0.0);

	for (int i = this->start_point, j = 0; i < this->end_point; i++, j++)
	{
		result_data[j] = (*this->data)[i] + _value;
	}

	return Tensor(this->shape, result_data);
}

Tensor Tensor::operator-(const double& _value) const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Tensor] Subtraction failed: cannot perform subtraction on empty Tensor.");
	}

	if (!Utils::IsValidData<double>({ _value }))
	{
		throw std::invalid_argument("[Tensor] Subtraction failed: invalid value.");
	}

	std::vector<double> result_data(this->volume, 0.0);

	for (int i = this->start_point, j = 0; i < this->end_point; i++, j++)
	{
		result_data[j] = (*this->data)[i] - _value;
	}

	return Tensor(this->shape, result_data);
}

Tensor Tensor::operator*(const double& _value) const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Tensor] Multiplication failed: cannot perform multiplication on empty Tensor.");
	}

	if (!Utils::IsValidData<double>({ _value }))
	{
		throw std::invalid_argument("[Tensor] Multiplication failed: invalid value.");
	}

	std::vector<double> result_data(this->volume, 0.0);

	for (int i = this->start_point, j = 0; i < this->end_point; i++, j++)
	{
		result_data[j] = (*this->data)[i] * _value;
	}

	return Tensor(this->shape, result_data);
}

Tensor Tensor::operator/(const double& _value) const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Tensor] Division failed: cannot perform division on empty Tensor.");
	}

	if (!Utils::IsValidData<double>({ _value }))
	{
		throw std::invalid_argument("[Tensor] Division failed: invalid value.");
	}

	if (std::abs(_value) < std::numeric_limits<double>::epsilon() * this->EPSILON_SCALE)
	{
		throw std::domain_error("[Tensor] Division failed: division by ~zero value detected.");
	}

	std::vector<double> result_data(this->volume, 0.0);

	for (int i = this->start_point, j = 0; i < this->end_point; i++, j++)
	{
		result_data[j] = (*this->data)[i] / _value;
	}

	return Tensor(this->shape, result_data);
}

Tensor Tensor::operator+(const Tensor& _tensor) const
{
	if (this->IsEmpty() || _tensor.IsEmpty())
	{
		throw std::runtime_error("[Tensor] Addition failed: cannot perform addition on empty Tensor(s).");
	}

	if (this->shape != _tensor.shape)
	{
		auto t1 = this->Broadcast(_tensor.shape);
		auto t2 = _tensor.Broadcast(this->shape);

		return (t1 + t2);
	}

	std::vector<double> result_data(this->volume, 0.0);

	for (int i = this->start_point, j = _tensor.start_point, k = 0; i < this->end_point && j < _tensor.end_point; i++, j++, k++)
	{
		result_data[k] = (*this->data)[i] + (*_tensor.data)[j];
	}

	return Tensor(this->shape, result_data);
}

Tensor Tensor::operator-(const Tensor& _tensor) const
{
	if (this->IsEmpty() || _tensor.IsEmpty())
	{
		throw std::runtime_error("[Tensor] Subtraction failed: cannot perform subtraction on empty Tensor(s).");
	}

	if (this->shape != _tensor.shape)
	{
		auto t1 = this->Broadcast(_tensor.shape);
		auto t2 = _tensor.Broadcast(this->shape);

		return (t1 - t2);
	}

	std::vector<double> result_data(this->volume, 0.0);

	for (int i = this->start_point, j = _tensor.start_point, k = 0; i < this->end_point && j < _tensor.end_point; i++, j++, k++)
	{
		result_data[k] = (*this->data)[i] - (*_tensor.data)[j];
	}

	return Tensor(this->shape, result_data);
}

Tensor Tensor::operator*(const Tensor& _tensor) const
{
	if (this->IsEmpty() || _tensor.IsEmpty())
	{
		throw std::runtime_error("[Tensor] Multiplication failed: cannot perform multiplication on empty Tensor(s).");
	}

	if (this->shape != _tensor.shape)
	{
		auto t1 = this->Broadcast(_tensor.shape);
		auto t2 = _tensor.Broadcast(this->shape);

		return (t1 * t2);
	}

	std::vector<double> result_data(this->volume, 0.0);

	for (int i = this->start_point, j = _tensor.start_point, k = 0; i < this->end_point && j < _tensor.end_point; i++, j++, k++)
	{
		result_data[k] = (*this->data)[i] * (*_tensor.data)[j];
	}

	return Tensor(this->shape, result_data);
}

Tensor Tensor::operator/(const Tensor& _tensor) const
{
	if (this->IsEmpty() || _tensor.IsEmpty())
	{
		throw std::runtime_error("[Tensor] Division failed: cannot perform division on empty Tensor(s).");
	}

	if (this->shape != _tensor.shape)
	{
		auto t1 = this->Broadcast(_tensor.shape);
		auto t2 = _tensor.Broadcast(this->shape);

		return (t1 / t2);
	}

	std::vector<double> result_data(this->volume, 0.0);

	for (int i = this->start_point, j = _tensor.start_point, k = 0; i < this->end_point && j < _tensor.end_point; i++, j++, k++)
	{
		if (std::abs((*_tensor.data)[j]) < std::numeric_limits<double>::epsilon() * this->EPSILON_SCALE)
		{
			throw std::domain_error("[Tensor] Division failed: division by ~zero value detected.");
		}
		result_data[k] = (*this->data)[i] / (*_tensor.data)[j];
	}

	return Tensor(this->shape, result_data);
}

void Tensor::operator+=(const double& _value)
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Tensor] Addition failed: cannot perform addition on empty Tensor.");
	}

	if (!Utils::IsValidData<double>({ _value }))
	{
		throw std::invalid_argument("[Tensor] Addition failed: invalid value.");
	}

	for (int i = this->start_point; i < this->end_point; i++)
	{
		(*this->data)[i] += _value;
	}
}

void Tensor::operator-=(const double& _value)
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Tensor] Subtraction failed: cannot perform subtraction on empty Tensor.");
	}

	if (!Utils::IsValidData<double>({ _value }))
	{
		throw std::invalid_argument("[Tensor] Subtraction failed: invalid value.");
	}

	for (int i = this->start_point; i < this->end_point; i++)
	{
		(*this->data)[i] -= _value;
	}
}

void Tensor::operator*=(const double& _value)
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Tensor] Multiplication failed: cannot perform multiplication on empty Tensor.");
	}

	if (!Utils::IsValidData<double>({ _value }))
	{
		throw std::invalid_argument("[Tensor] Multiplication failed: invalid value.");
	}

	for (int i = this->start_point; i < this->end_point; i++)
	{
		(*this->data)[i] *= _value;
	}
}

void Tensor::operator/=(const double& _value)
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Tensor] Division failed: cannot perform division on empty Tensor.");
	}

	if (!Utils::IsValidData<double>({ _value }))
	{
		throw std::invalid_argument("[Tensor] Division failed: invalid value.");
	}

	if (std::abs(_value) < std::numeric_limits<double>::epsilon() * this->EPSILON_SCALE)
	{
		throw std::domain_error("[Tensor] Division failed: division by ~zero value detected.");
	}

	for (int i = this->start_point; i < this->end_point; i++)
	{
		(*this->data)[i] /= _value;
	}
}

void Tensor::operator+=(const Tensor& _tensor)
{
	if (this->IsEmpty() || _tensor.IsEmpty())
	{
		throw std::runtime_error("[Tensor] Addition failed: cannot perform addition on empty Tensor(s).");
	}

	if (this->IsScalar() && _tensor.IsScalar())
	{
		(*this->data)[this->start_point] += (*_tensor.data)[_tensor.start_point];
		return;
	}

	*this = this->Broadcast(_tensor.shape);
	auto t2 = _tensor.Broadcast(this->shape);

	for (int i = this->start_point, j = _tensor.start_point; i < this->end_point && j < _tensor.end_point; i++, j++)
	{
		(*this->data)[i] += (*_tensor.data)[j];
	}
}

void Tensor::operator-=(const Tensor& _tensor)
{
	if (this->IsEmpty() || _tensor.IsEmpty())
	{
		throw std::runtime_error("[Tensor] Subtraction failed: cannot perform subtraction on empty Tensor(s).");
	}

	if (this->IsScalar() && _tensor.IsScalar())
	{
		(*this->data)[this->start_point] -= (*_tensor.data)[_tensor.start_point];
		return;
	}

	*this = this->Broadcast(_tensor.shape);
	auto t2 = _tensor.Broadcast(this->shape);

	for (int i = this->start_point, j = _tensor.start_point; i < this->end_point && j < _tensor.end_point; i++, j++)
	{
		(*this->data)[i] -= (*_tensor.data)[j];
	}
}

void Tensor::operator*=(const Tensor& _tensor)
{
	if (this->IsEmpty() || _tensor.IsEmpty())
	{
		throw std::runtime_error("[Tensor] Multiplication failed: cannot perform multiplication on empty Tensor(s).");
	}

	if (this->IsScalar() && _tensor.IsScalar())
	{
		(*this->data)[this->start_point] *= (*_tensor.data)[_tensor.start_point];
		return;
	}

	*this = this->Broadcast(_tensor.shape);
	auto t2 = _tensor.Broadcast(this->shape);

	for (int i = this->start_point, j = _tensor.start_point; i < this->end_point && j < _tensor.end_point; i++, j++)
	{
		(*this->data)[i] *= (*_tensor.data)[j];
	}
}

void Tensor::operator/=(const Tensor& _tensor)
{
	if (this->IsEmpty() || _tensor.IsEmpty())
	{
		throw std::runtime_error("[Tensor] Division failed: cannot perform division on empty Tensor(s).");
	}

	if (this->IsScalar() && _tensor.IsScalar())
	{
		if (std::abs((*_tensor.data)[_tensor.start_point]) < std::numeric_limits<double>::epsilon() * this->EPSILON_SCALE)
		{
			throw std::domain_error("[Tensor] Division failed: division by ~zero value detected.");
		}

		(*this->data)[this->start_point] /= (*_tensor.data)[_tensor.start_point];
		return;
	}

	*this = this->Broadcast(_tensor.shape);
	auto t2 = _tensor.Broadcast(this->shape);

	for (int i = _tensor.start_point; i < _tensor.end_point; i++)
	{
		if (std::abs((*_tensor.data)[i]) < std::numeric_limits<double>::epsilon() * this->EPSILON_SCALE)
		{
			throw std::domain_error("[Tensor] Division failed: division by ~zero value detected.");
		}
	}

	for (int i = this->start_point, j = _tensor.start_point; i < this->end_point && j < _tensor.end_point; i++, j++)
	{
		(*this->data)[i] /= (*_tensor.data)[j];
	}
}

// ========================================
// Tensor Reshaping Method(s)
// ========================================
Tensor Tensor::Reshape(const std::vector<int>& _new_shape) const
{
	if (!Utils::IsAllPositive(_new_shape))
	{
		throw std::invalid_argument("[Tensor] Reshape failed: all shape dimensions must be > 0.");
	}

	if (this->volume != Utils::ShapeToVolume(_new_shape))
	{
		throw std::invalid_argument("[Tensor] Reshape failed: new volume shape volume mismatch with current Tensor volume.");
	}

	auto start_ptr = this->data->begin() + this->start_point;
	auto end_ptr = this->data->begin() + this->end_point;

	std::vector<double> data_copy(start_ptr, end_ptr);

	return Tensor(_new_shape, (*this->data));
}

Tensor Tensor::ExpandRank(const int& _axis) const
{
	if (_axis < 0 || _axis > this->rank)
	{
		throw std::out_of_range("[Tensor] Expanding Rank failed: axis out of bound for rank expanding.");
	}

	std::vector<int> new_shape = this->shape;
	new_shape.insert((new_shape.begin() + _axis), 1);

	return this->Reshape(new_shape);
}

Tensor Tensor::Flatten(const int& _axis_from, const int& _axis_upto) const
{
	if (this->rank <= 1)
	{
		throw std::runtime_error("[Tensor] Flatten failed: cannot flatten a rank-0 or rank-1 Tensor.");
	}

	if (_axis_from < 0 || _axis_upto > this->rank)
	{
		throw std::out_of_range("[Tensor] Flatten failed: index values out of bounds.");
	}

	if (_axis_from >= _axis_upto)
	{
		throw std::invalid_argument("[Tensor] Flatten failed: axis_from must be less than axis_upto.");
	}

	auto start_ptr = this->shape.begin() + _axis_from;
	auto end_ptr = this->shape.begin() + _axis_upto;

	int flat_volume = Utils::ShapeToVolume(std::vector<int>(start_ptr, end_ptr));

	std::vector<int> new_shape(this->shape.begin(), start_ptr);
	new_shape.push_back(flat_volume);
	new_shape.insert(new_shape.end(), end_ptr, this->shape.end());

	auto data_start = this->data->begin() + this->start_point;
    auto data_end = this->data->begin() + this->end_point;

    std::vector<double> data_copy(data_start, data_end);

	return Tensor(new_shape, data_copy);
}

// ========================================
// Tensor Slicing Method(s)
// ========================================
Tensor Tensor::Slice(const int& _axis, const int& _index) const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Tensor] Slicing failed: cannot slice an empty Tensor.");
	}

	if (this->IsScalar())
	{
		throw std::runtime_error("[Tensor] Slicing failed: cannot slice a scalar Tensor.");
	}

	if (_axis < 0 || _axis >= this->rank)
	{
		throw std::out_of_range("[Tensor] Slicing failed: axis out of bound for slicing.");
	}

	if (_index < 0 || _index >= this->shape[_axis])
	{
		throw std::out_of_range("[Tensor] Slicing failed: index out of bound in specified axis.");
	}

	std::vector<int> new_shape = this->shape;
	new_shape.erase(new_shape.begin() + _axis);

	int new_volume = Utils::ShapeToVolume(new_shape);

	std::vector<double> new_data;
	new_data.reserve(new_volume);

	int cursor = this->start_point + (_index * this->strides[_axis]);
	int bucket = this->strides[_axis];
	int jump = (_axis > 0) ? this->strides[_axis - 1] : this->volume;

	while (cursor < this->end_point)
	{
		auto start_ptr = this->data->begin() + cursor;
		auto end_ptr = start_ptr + bucket;

		new_data.insert(new_data.end(), start_ptr, end_ptr);

		cursor += jump;
	}

	return Tensor(new_shape, new_data);
}

Tensor Tensor::Slice(const int& _axis, const int& _index_from, const int& _index_upto) const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Tensor] Slicing failed: cannot slice an empty Tensor.");
	}

	if (this->IsScalar())
	{
		throw std::runtime_error("[Tensor] Slicing failed: cannot slice a scalar Tensor.");
	}

	if (_axis < 0 || _axis >= this->rank)
	{
		throw std::out_of_range("[Tensor] Slicing failed: axis out of bound for slicing.");
	}

	if (_index_from < 0 || _index_upto > this->shape[_axis])
	{
		throw std::out_of_range("[Tensor] Slicing failed: index out of bound in specified axis.");
	}

	if (_index_from >= _index_upto)
	{
		throw std::invalid_argument("[Tensor] Slicing failed: index_from must be less than index_upto.");
	}

	std::vector<Tensor> slices(_index_upto - _index_from);

	for (int index = _index_from, i = 0; index < _index_upto; index++, i++)
	{
		slices[i] = this->Slice(_axis, index);
	}

	return Tensor::Stack(slices, _axis);
}

// ========================================
// Tensor Appending Method(s)
// ========================================
void Tensor::Append(const Tensor& _tensor, const int& _axis)
{
	this->UniqueData();

	if (_axis < -1 || _axis >= this->rank)
	{
		throw std::out_of_range("[Tensor] Appending failed: axis out of bound for append operation.");
	}

	if (_tensor.rank != (this->rank - 1))
	{
		throw std::invalid_argument("[Tensor] Appending failed: sub Tensor rank must be one less than base Tensor.");
	}

	int axis = _axis;
	for (int i = (this->rank - 1), j = (_tensor.rank - 1); i >= 0 && j >= 0; )
	{
		if (this->shape[i] == _tensor.shape[j])
		{
			i--;
			j--;
		}
		else if (axis == -1 || axis == i)
		{
			axis = i;
			i--;
		}
		else
		{
			throw std::invalid_argument("[Tensor] Appending failed: shape not compatible for appending in specified axis.");
		}
	}

	axis = (axis <= -1) ? 0 : axis;

	std::vector<int> result_shape = this->shape;
	result_shape[axis] += 1;

	if (Utils::IsVolumeOverflow(result_shape))
	{
		throw std::overflow_error("[Tensor] Appending failed: shape too large, potential overflow.");
	}

	std::vector<double> merged_data;

	int vol_1 = (axis == 0) ? this->volume : this->strides[axis - 1];
	int vol_2 = this->strides[axis];

	int op = this->volume / (this->strides[axis] * this->shape[axis]);

	for(int i = 0; i < op; i++)
	{
		auto this_start_ptr = this->data->begin() + this->start_point + (i * vol_1);
		auto this_end_ptr = this_start_ptr + vol_1;

		merged_data.insert(merged_data.end(), this_start_ptr, this_end_ptr);

		auto other_start_ptr = (*_tensor.data).begin() + _tensor.start_point + (i * vol_2);
		auto other_end_ptr = other_start_ptr + vol_2;

		merged_data.insert(merged_data.end(), other_start_ptr, other_end_ptr);
	}

	this->data = std::make_shared<std::vector<double>>(merged_data);

	this->shape = result_shape;
	this->strides = Utils::ShapeToStrides(this->shape);

	this->volume = Utils::ShapeToVolume(this->shape);

	this->start_point = 0;
	this->end_point = this->volume;
}

void Tensor::Insert(const Tensor& _tensor, const int& _axis, const int& _index)
{
	this->UniqueData();

	if (_axis < -1 || _axis >= this->rank)
	{
		throw std::out_of_range("[Tensor] Inserting failed: axis out of bound for insert operation.");
	}

	if (_tensor.rank != (this->rank - 1))
	{
		throw std::invalid_argument("[Tensor] Inserting failed: sub Tensor rank must be one less than base Tensor.");
	}

	int axis = _axis;
	for (int i = (this->rank - 1), j = (_tensor.rank - 1); i >= 0 && j >= 0; )
	{
		if (this->shape[i] == _tensor.shape[j])
		{
			i--;
			j--;
		}
		else if (axis == -1 || axis == i)
		{
			axis = i;
			i--;
		}
		else
		{
			throw std::invalid_argument("[Tensor] Inserting failed: shape not compatible for inserting in specified axis.");
		}
	}

	axis = (axis <= -1) ? 0 : axis;

	if (_index < 0 || _index > this->shape[axis])
	{
		throw std::out_of_range("[Tensor] Inserting failed: index out of bound for insert operation.");
	}

	std::vector<int> result_shape = this->shape;
	result_shape[axis] += 1;

	if (Utils::IsVolumeOverflow(result_shape))
	{
		throw std::overflow_error("[Tensor] Inserting failed: shape too large, potential overflow.");
	}

	std::vector<double> merged_data;

	int vol_1 = (axis == 0) ? this->volume : this->strides[axis - 1];
	int vol_11 = (vol_1 * _index) / this->shape[axis];
	int vol_12 = vol_1 - vol_11;

	int vol_2 = this->strides[axis];

	int op = this->volume / (this->strides[axis] * this->shape[axis]);

	for (int i = 0; i < op; i++)
	{
		auto this_start_ptr = this->data->begin() + this->start_point + (i * vol_1);
		auto this_end_ptr = this_start_ptr + vol_11;

		merged_data.insert(merged_data.end(), this_start_ptr, this_end_ptr);

		auto other_start_ptr = (*_tensor.data).begin() + _tensor.start_point + (i * vol_2);
		auto other_end_ptr = other_start_ptr + vol_2;

		merged_data.insert(merged_data.end(), other_start_ptr, other_end_ptr);

		this_start_ptr = this_end_ptr;
		this_end_ptr = this_start_ptr + vol_12;

		merged_data.insert(merged_data.end(), this_start_ptr, this_end_ptr);
	}

	this->data = std::make_shared<std::vector<double>>(merged_data);

	this->shape = result_shape;
	this->strides = Utils::ShapeToStrides(this->shape);

	this->volume = Utils::ShapeToVolume(this->shape);

	this->start_point = 0;
	this->end_point = this->volume;
}

Tensor Tensor::Concat(const std::vector<Tensor>& _tensors, const int& _axis)
{
	if (_axis < -1)
	{
		throw std::invalid_argument("[Tensor] Concatenation failed: invalid axis for concatenation.");
	}

	if (_tensors.empty())
	{
		throw std::invalid_argument("[Tensor] Concatenation failed: empty array of Tensor.");
	}

	int axis = _axis;
	for (int i = 1; i < _tensors.size(); i++)
	{
		if (_tensors[i].rank != _tensors[i - 1].rank)
		{
			throw std::invalid_argument("[Tensor] Concatenation failed: rank mismatch found in Tensors.");
		}

		if (axis >= _tensors[i].rank)
		{
			throw std::out_of_range("[Tensor] Concatenation failed: axis out of bounds for concatenation.");
		}

		for (int d = 0; d < _tensors[i].rank; d++)
		{
			if ((_tensors[i].shape[d] != _tensors[i - 1].shape[d]) && axis != d)
			{
				if (axis != -1)
				{
					throw std::invalid_argument("[Tensor] Concatenation failed: shape of Tensors not compatible for concatenation.");
				}
				axis = d;
			}
		}
	}

	axis = (axis <= -1) ? 0 : axis;

	int concat_dim_size = 0;
	for (auto& tensor : _tensors)
	{
		concat_dim_size += tensor.shape[axis];
	}

	std::vector<int> concat_shape = _tensors[0].shape;
	concat_shape[axis] = concat_dim_size;

	if (Utils::IsVolumeOverflow(concat_shape))
	{
		throw std::overflow_error("[Tensor] Concatenation failed: shape too large, potential overflow.");
	}

	int rank = _tensors[0].rank;
	int concat_volume = Utils::ShapeToVolume(concat_shape);
	std::vector<double> concat_data(concat_volume, 0.0);

	int lower_volume = 1;
	for (int i = axis + 1; i < rank; ++i)
	{
		lower_volume *= concat_shape[i];
	}

	int offset = 0;
	for (auto& tensor : _tensors)
	{
		int upper_volume = tensor.volume / (lower_volume * tensor.shape[axis]);

		for (int index = tensor.start_point; index < tensor.end_point; index += lower_volume)
		{
			std::vector<int> tensor_index = Utils::TensorIndex(tensor.shape, index);
			tensor_index[axis] += offset;

			int flat_index = Utils::FlatIndex(concat_shape, tensor_index);

			auto start_ptr = tensor.data->begin() + index;
			auto end_ptr = tensor.data->begin() + index + lower_volume;
			auto loc_ptr = concat_data.begin() + flat_index;

			std::copy(start_ptr, end_ptr, loc_ptr);
		}
		offset += tensor.shape[axis];
	}

	return Tensor(concat_shape, concat_data);
}

Tensor Tensor::Stack(const std::vector<Tensor>& _tensors, const int& _axis)
{
	if (_axis < 0)
	{
		throw std::out_of_range("[Tensor] Stacking failed: invalid axis to stack - found negative axis.");
	}

	if (_tensors.empty())
	{
		throw std::invalid_argument("[Tensor] Stacking failed: empty array of Tensor.");
	}

	for (int i = 1; i < _tensors.size(); i++)
	{
		if (_tensors[i].shape != _tensors[i - 1].shape)
		{
			throw std::invalid_argument("[Tensor] Stacking failed: Tensor shape mismatch for stacking.");
		}
	}

	if (_axis > _tensors[0].rank)
	{
		throw std::out_of_range("[Tensor] Stacking failed: axis out of bounds for stacking.");
	}

	std::vector<Tensor> expanded_tensors(_tensors.size());

	for (int i = 0; i < _tensors.size(); i++)
	{
		expanded_tensors[i] = _tensors[i].ExpandRank(_axis);
	}

	return Tensor::Concat(expanded_tensors, _axis);
}

Tensor Tensor::Pad(const int& _axis, const int& _pad_before_size, const int& _pad_after_size, const double& _value) const
{
	if (_axis < 0 || _axis >= this->rank)
	{
		throw std::out_of_range("[Tensor] Padding failed: axis out of bounds for padding.");
	}

	if (!Utils::IsValidData<double>({ _value }))
	{
		throw std::invalid_argument("[Tensor] Padding failed: invalid value.");
	}

	if (_pad_before_size < 0)
	{
		throw std::invalid_argument("[Tensor] Padding failed: pad_before_size value cannot be less than 0.");
	}
	if (_pad_after_size < 0)
	{
		throw std::invalid_argument("[Tensor] Padding failed: pad_after_size value cannot be less than 0.");
	}

	std::vector<int> new_shape = this->shape;
	new_shape[_axis] += (_pad_before_size + _pad_after_size);

	if (Utils::IsVolumeOverflow(new_shape))
	{
		throw std::overflow_error("[Tensor] Padding failed: shape too large, potential overflow.");
	}

	Tensor padded_tensor = *this;

	if (_pad_before_size)
	{
		std::vector<int> pad_shape = this->shape;
		pad_shape[_axis] = _pad_before_size;

		Tensor pad(pad_shape, _value);

		padded_tensor = Tensor::Concat({ pad, padded_tensor }, _axis);
	}
	if (_pad_after_size)
	{
		std::vector<int> pad_shape = this->shape;
		pad_shape[_axis] = _pad_after_size;

		Tensor pad(pad_shape, _value);

		padded_tensor = Tensor::Concat({ padded_tensor, pad }, _axis);
	}

	return padded_tensor;
}

Tensor Tensor::Tile(const std::vector<int>& _repetitions) const
{
	if (_repetitions.size() != this->rank)
	{
		throw std::invalid_argument("[Tensor] Tiling failed: array of repetition size mismatch with Tensor shape size.");
	}

	if (!Utils::IsAllPositive(_repetitions))
	{
		throw std::invalid_argument("[Tensor] Tiling failed: repetitions must be all positive.");
	}

	if (Utils::IsVolumeOverflow(_repetitions))
	{
		throw std::overflow_error("[Tensor] Tiling failed: end shape too large, potential volume overflow.");
	}

	long long total_volume = this->volume;
	total_volume *= Utils::ShapeToVolume(_repetitions);

	if (total_volume > INT_MAX)
	{
		throw std::overflow_error("[Tensor] Tiling failed: end shape too large, potential volume overflow.");
	}

	Tensor result_tensor = *this;

	for (int axis = (this->rank - 1); axis >= 0; axis--)
	{
		if (_repetitions[axis] == 1)
		{
			continue;
		}

		std::vector<Tensor> tiles(_repetitions[axis], result_tensor);
		result_tensor = Tensor::Concat(tiles, axis);
	}

	return result_tensor;
}

// ========================================
// Tensor Broadcasting Method
// ========================================
Tensor Tensor::Broadcast(const std::vector<int>& _shape) const
{
	if (_shape.empty() || !Utils::IsAllPositive(_shape))
	{
		throw std::invalid_argument("[Tensor] Broadcast failed: invalid shape for broadcasting.");
	}

	if (this->rank == 0)
	{
		if (this->volume == 0)
		{
			throw std::invalid_argument("[Tensor] Broadcast failed: cannot broadcast an empty Tensor.");
		}
		return Tensor(_shape, (*this->data)[this->start_point]);
	}

	if (this->shape == _shape)
	{
		return *this;
	}

	if (!Utils::IsBroadcastCompatible(this->shape, _shape))
	{
		throw std::invalid_argument("[Tensor] Broadcast failed: shapes are not compatible for broadcasting.");
	}

	std::vector<int> broadcast_shape = Utils::BroadcastShape(this->shape, _shape);

	int broadcast_rank = broadcast_shape.size();
	int rank_diff = abs(this->rank - broadcast_rank);

	int broadcast_volume = Utils::ShapeToVolume(broadcast_shape);

	std::vector<double> broadcast_data(broadcast_volume, 0.0);

	for (int i = 0; i < broadcast_volume; i++)
	{
		std::vector<int> broadcast_tensor_index = Utils::TensorIndex(broadcast_shape, i);
		std::vector<int> tensor_index(this->rank, 0);

		for (int j = 0; j < this->rank; j++)
		{
			int offset_dimension = rank_diff + j;
			int index = broadcast_tensor_index[offset_dimension];

			tensor_index[j] = (this->shape[j] == 1) ? 0 : index;
		}

		int flat_index = Utils::FlatIndex(this->shape, tensor_index);
		broadcast_data[i] = (*this->data)[this->start_point + flat_index];
	}

	return Tensor(broadcast_shape, broadcast_data);
}

// ========================================
// Tensor Transpose Method
// ========================================
Tensor Tensor::Transpose(const std::vector<int>& _permutation) const
{
	if (_permutation.size() != this->rank)
	{
		throw std::invalid_argument("[Tensor] Transposing failed: size mismatch between permutation and Tensor's rank.");
	}

	if (Utils::IsAnyNegative(_permutation))
	{
		throw std::invalid_argument("[Tensor] Transposing failed: negative value(s) found in permutation array.");
	}

	if (!Utils::IsUpperBounded(_permutation, this->rank, true))
	{
		throw std::invalid_argument("[Tensor] Tranposing failed: values of permutation found >= rank of Tensor.");
	}

	if (!Utils::IsAllUnique(_permutation))
	{
		throw std::invalid_argument("[Tensor] Transposing failed: duplicate values found in permutation array.");
	}

	std::vector<int> transposed_shape = Utils::Permute(this->shape, _permutation);
	std::vector<double> transposed_data(this->volume);

	for (int i = this->start_point; i < this->end_point; i++)
	{
		auto tensor_index = Utils::TensorIndex(this->shape, i);
		auto transposed_index = Utils::Permute(tensor_index, _permutation);

		int flat_index = Utils::FlatIndex(transposed_shape, transposed_index);
		transposed_data[flat_index] = (*this->data)[i];
	}

	return Tensor(transposed_shape, transposed_data);
}

// ========================================
// Tensor Dot Product Method(s)
// ========================================
Tensor Tensor::MatMul(const Tensor& _tensor_1, const Tensor& _tensor_2)
{
	if (_tensor_1.rank == 0 || _tensor_2.rank == 0)
	{
		throw std::invalid_argument("[Tensor] Matrix Multiplication failed: rank of Tensor(s) must be > 0.");
	}

	Tensor tensor_1 = _tensor_1;
	Tensor tensor_2 = _tensor_2;

	if (tensor_1.rank == 1)
	{
		tensor_1 = tensor_1.ExpandRank(0);
	}

	if (tensor_2.rank == 1)
	{
		tensor_2 = tensor_2.ExpandRank(tensor_2.rank);
	}

	std::vector<int> batch_shape_1(tensor_1.shape.begin(), (tensor_1.shape.end() - 2));
	std::vector<int> batch_shape_2(tensor_2.shape.begin(), (tensor_2.shape.end() - 2));
	std::vector<int> matrix_shape_1((tensor_1.shape.end() - 2), tensor_1.shape.end());
	std::vector<int> matrix_shape_2((tensor_2.shape.end() - 2), tensor_2.shape.end());

	if (matrix_shape_1[1] != matrix_shape_2[0])
	{
		throw std::invalid_argument("[Tensor] Matrix Multiplication failed: inner dimensions must match (got "
			+ std::to_string(matrix_shape_1[1]) + " and " + std::to_string(matrix_shape_2[0]) + ").");
	}

	std::vector<int> batch_shape = Utils::BroadcastShape(batch_shape_1, batch_shape_2);

	std::vector<int> new_shape_1 = batch_shape;
	new_shape_1.insert(new_shape_1.end(), matrix_shape_1.begin(), matrix_shape_1.end());

	std::vector<int> new_shape_2 = batch_shape;
	new_shape_2.insert(new_shape_2.end(), matrix_shape_2.begin(), matrix_shape_2.end());

	if (tensor_1.shape != new_shape_1)
	{
		tensor_1 = tensor_1.Broadcast(new_shape_1);
	}
	if (tensor_2.shape != new_shape_2)
	{
		tensor_2 = tensor_2.Broadcast(new_shape_2);
	}

	int mat_volume_1 = Utils::ShapeToVolume(matrix_shape_1);
	int mat_volume_2 = Utils::ShapeToVolume(matrix_shape_2);

	std::vector<int> result_shape = batch_shape;
	result_shape.push_back(matrix_shape_1[0]);
	result_shape.push_back(matrix_shape_2[1]);

	std::vector<double> result_data;
	result_data.reserve(Utils::ShapeToVolume(result_shape));

	for (int i = tensor_1.start_point, j = tensor_2.start_point; i < tensor_1.end_point && j < tensor_2.end_point; i += mat_volume_1, j += mat_volume_2)
	{
		auto start_ptr_1 = (tensor_1.data)->begin() + i;
		auto end_ptr_1 = start_ptr_1 + mat_volume_1;

		auto start_ptr_2 = (tensor_2.data)->begin() + j;
		auto end_ptr_2 = start_ptr_2 + mat_volume_2;

		std::vector<double> data_1(start_ptr_1, end_ptr_1);
		std::vector<double> data_2(start_ptr_2, end_ptr_2);

		auto matrix_1 = Utils::VectorToMatrix(data_1, { matrix_shape_1[0], matrix_shape_1[1] });
		auto matrix_2 = Utils::VectorToMatrix(data_2, { matrix_shape_2[0], matrix_shape_2[1] });

		auto result_matrix = Utils::StandardMatrixMultiply(matrix_1, matrix_2);
		auto vec = Utils::MatrixToVector(result_matrix);

		result_data.insert(result_data.end(), vec.begin(), vec.end());
	}

	return Tensor(result_shape, result_data);
}

Tensor Tensor::MatMul(const Tensor& _tensor) const
{
	return Tensor::MatMul(*this, _tensor);
}

Tensor Tensor::TensorDot(const Tensor& _tensor_1, const Tensor& _tensor_2, const std::vector<int>& _contract_axes_1, const std::vector<int>& _contract_axes_2)
{
	if (!Utils::IsBounded(_contract_axes_1, _tensor_1.rank, -1, true))
	{
		throw std::out_of_range("[Tensor] Tensor-Dot failed: value(s) of contract_axes_1 are out of bounds.");
	}

	if (!Utils::IsBounded(_contract_axes_2, _tensor_2.rank, -1, true))
	{
		throw std::out_of_range("[Tensor] Tensor-Dot failed: value(s) of contract_axes_2 are out of bounds.");
	}

	if (!Utils::IsAllUnique(_contract_axes_1))
	{
		throw std::invalid_argument("[Tensor] Tensor-Dot failed: repeating values found in contract_axes_1 argument.");
	}

	if (!Utils::IsAllUnique(_contract_axes_2))
	{
		throw std::invalid_argument("[Tensor] Tensor-Dot failed: repeating values found in contract_axes_2 argument.");
	}

	if (_contract_axes_1.size() != _contract_axes_2.size())
	{
		throw std::invalid_argument("[Tensor] Tensor-Dot failed: number of contracting axes must match.");
	}

	int contract_volume_1 = 1;
	int contract_volume_2 = 1;

	for (size_t i = 0; i < _contract_axes_1.size(); i++)
	{
		int dim_1 = _tensor_1.shape[_contract_axes_1[i]];
		int dim_2 = _tensor_2.shape[_contract_axes_2[i]];

		if (dim_1 != dim_2)
		{
			throw std::invalid_argument("[Tensor] Tensor-Dot failed: contracted axis dimensions must match (got "
				+ std::to_string(dim_1) + " and " + std::to_string(dim_2) + ").");
		}

		contract_volume_1 *= dim_1;
		contract_volume_2 *= dim_2;
	}

	auto permutation_1 = Utils::FindRangeComplement(_contract_axes_1, { 0, _tensor_1.rank });
	permutation_1.insert(permutation_1.end(), _contract_axes_1.begin(), _contract_axes_1.end());

	auto remaining_2 = Utils::FindRangeComplement(_contract_axes_2, { 0, _tensor_2.rank });
	auto permutation_2 = _contract_axes_2;
	permutation_2.insert(permutation_2.end(), remaining_2.begin(), remaining_2.end());

	int batch_1 = _tensor_1.volume / contract_volume_1;
	int batch_2 = _tensor_2.volume / contract_volume_2;

	Tensor tensor_1 = _tensor_1.Transpose(permutation_1).Reshape({ batch_1, contract_volume_1 });
	Tensor tensor_2 = _tensor_2.Transpose(permutation_2).Reshape({ contract_volume_2, batch_2 });

	Tensor dot_product = Tensor::MatMul(tensor_1, tensor_2);

	auto shape_1 = Utils::Permute(_tensor_1.shape, permutation_1);
	auto shape_2 = Utils::Permute(_tensor_2.shape, permutation_2);

	std::vector<int> tensordot_shape;
	tensordot_shape.insert(tensordot_shape.end(), shape_1.begin(), (shape_1.end() - _contract_axes_1.size()));
	tensordot_shape.insert(tensordot_shape.end(), (shape_2.begin() + _contract_axes_2.size()), shape_2.end());

	Tensor result = dot_product.Reshape(tensordot_shape);

	return result;
}

// ========================================
// Tensor Convolution Method(s)
// ========================================
Tensor Tensor::Convolve(const Tensor& _filter, const std::vector<int>& _strides, const std::vector<int>& _padding)
{
	if (_strides.size() != this->rank)
	{
		throw std::invalid_argument("[Tensor] Convolution failed: stride size mismatch with Tensor's rank.");
	}

	if (_padding.size() != this->rank)
	{
		throw std::invalid_argument("[Tensor] Convolution failed: padding size mismatch with Tensor's rank.");
	}

	if (!Utils::IsAllPositive(_strides))
	{
		throw std::invalid_argument("[Tensor] Convolution failed: stride values must be positive.");
	}

	if (Utils::IsAnyNegative(_padding))
	{
		throw std::invalid_argument("[Tensor] Convolution failed: found negative padding value. padding value(s) should be >= 0.");
	}

	std::vector<int> padded_shape(this->rank);

	for (int i = 0; i < this->rank; i++)
	{
		padded_shape[i] = this->shape[i] + (2 * _padding[i]);
	}

	if (Utils::IsVolumeOverflow(padded_shape))
	{
		throw std::overflow_error("[Tensor] Convolution failed: shape too large, potential overflow.");
	}

	Tensor padded_tensor = *this;
	for (int i = 0; i < this->rank; i++)
	{
		padded_tensor = padded_tensor.Pad(i, _padding[i], _padding[i], 0.0);
	}

	if (!Utils::IsConvolveCompatible(padded_tensor.shape, _filter.shape))
	{
		throw std::invalid_argument("[Tensor] Convolution failed: kernel shape is not compatible with Tensor for convolution.");
	}

	std::vector<int> broadcasted_shape((padded_tensor.rank - _filter.rank), 1);
	broadcasted_shape.insert(broadcasted_shape.end(), _filter.shape.begin(), _filter.shape.end());

	Tensor broadcasted_filter = _filter.Reshape(broadcasted_shape);

	auto convolved_shape = Utils::ConvolvedFeatureShape(padded_tensor.shape, broadcasted_filter.shape, _strides);
	int convolved_volume = Utils::ShapeToVolume(convolved_shape);

	std::vector<double> convolved_data(convolved_volume, 0.0);

	for (int i = 0; i < convolved_volume; i++)
	{
		double sum = 0.0;
		auto cnv_tensor_index = Utils::TensorIndex(convolved_shape, i);

		for (int k = broadcasted_filter.start_point; k < broadcasted_filter.end_point; k++)
		{
			auto filter_index = Utils::TensorIndex(broadcasted_filter.shape, (k - broadcasted_filter.start_point));
			auto main_tensor_index = Utils::ScaleNShift(cnv_tensor_index, _strides, filter_index);

			int flat_index = Utils::FlatIndex(padded_tensor.shape, main_tensor_index) + padded_tensor.start_point;
			sum += ((*padded_tensor.data)[flat_index] * (*broadcasted_filter.data)[k]);
		}

		convolved_data[i] = sum;
	}

	return Tensor(convolved_shape, convolved_data);
}

// ========================================
// Tensor Pooling Method(s)
// ========================================
Tensor Tensor::MaxPool(const std::vector<int>& _pool_shape, const std::vector<int>& _strides)
{
	if (!Utils::IsConvolveCompatible(this->shape, _pool_shape))
	{
		throw std::invalid_argument("[Tensor] Max Pooling failed: kernel shape is not compatible with Tensor for max pooling.");
	}

	std::vector<int> broadcasted_pool_shape((this->rank - _pool_shape.size()), 1);
	broadcasted_pool_shape.insert(broadcasted_pool_shape.end(), _pool_shape.begin(), _pool_shape.end());

	std::vector<int> pool_strides = (_strides.empty()) ? broadcasted_pool_shape : _strides;

	if (pool_strides.size() != this->rank)
	{
		throw std::invalid_argument("[Tensor] Max Pooling failed: stride size mismatch with Tensor's rank.");
	}

	if (!Utils::IsAllPositive(pool_strides))
	{
		throw std::invalid_argument("[Tensor] Max Pooling failed: stride values must be positive.");
	}

	auto pool_volume = Utils::ShapeToVolume(broadcasted_pool_shape);

	auto feature_shape = Utils::ConvolvedFeatureShape(this->shape, broadcasted_pool_shape, pool_strides);
	int feature_volume = Utils::ShapeToVolume(feature_shape);

	std::vector<double> feature_data(feature_volume, 0.0);

	for (int i = 0; i < feature_volume; i++)
	{
		double max_value = std::numeric_limits<double>::lowest();
		auto feature_tensor_index = Utils::TensorIndex(feature_shape, i);

		for (int j = 0; j < pool_volume; j++)
		{
			auto pool_index = Utils::TensorIndex(broadcasted_pool_shape, j);
			auto main_tensor_index = Utils::ScaleNShift(feature_tensor_index, pool_strides, pool_index);

			int flat_index = Utils::FlatIndex(this->shape, main_tensor_index) + this->start_point;
			max_value = std::max(max_value, (*this->data)[flat_index]);
		}

		feature_data[i] = max_value;
	}

	return Tensor(feature_shape, feature_data);
}

Tensor Tensor::MinPool(const std::vector<int>& _pool_shape, const std::vector<int>& _strides)
{
	if (!Utils::IsConvolveCompatible(this->shape, _pool_shape))
	{
		throw std::invalid_argument("[Tensor] Min Pooling failed: kernel shape is not compatible with Tensor for min pooling.");
	}

	std::vector<int> broadcasted_pool_shape((this->rank - _pool_shape.size()), 1);
	broadcasted_pool_shape.insert(broadcasted_pool_shape.end(), _pool_shape.begin(), _pool_shape.end());

	std::vector<int> pool_strides = (_strides.empty()) ? broadcasted_pool_shape : _strides;

	if (pool_strides.size() != this->rank)
	{
		throw std::invalid_argument("[Tensor] Max Pooling failed: stride size mismatch with Tensor's rank.");
	}

	if (!Utils::IsAllPositive(pool_strides))
	{
		throw std::invalid_argument("[Tensor] Min Pooling failed: stride values must be positive.");
	}

	auto pool_volume = Utils::ShapeToVolume(broadcasted_pool_shape);

	auto feature_shape = Utils::ConvolvedFeatureShape(this->shape, broadcasted_pool_shape, pool_strides);
	int feature_volume = Utils::ShapeToVolume(feature_shape);

	std::vector<double> feature_data(feature_volume, 0.0);

	for (int i = 0; i < feature_volume; i++)
	{
		double min_value = std::numeric_limits<double>::max();
		auto feature_tensor_index = Utils::TensorIndex(feature_shape, i);

		for (int j = 0; j < pool_volume; j++)
		{
			auto pool_index = Utils::TensorIndex(broadcasted_pool_shape, j);
			auto main_tensor_index = Utils::ScaleNShift(feature_tensor_index, pool_strides, pool_index);

			int flat_index = Utils::FlatIndex(this->shape, main_tensor_index) + this->start_point;
			min_value = std::min(min_value, (*this->data)[flat_index]);
		}

		feature_data[i] = min_value;
	}

	return Tensor(feature_shape, feature_data);
}

Tensor Tensor::AvgPool(const std::vector<int>& _pool_shape, const std::vector<int>& _strides)
{
	if (!Utils::IsConvolveCompatible(this->shape, _pool_shape))
	{
		throw std::invalid_argument("[Tensor] Average Pooling failed: kernel shape is not compatible with Tensor for average pooling.");
	}

	std::vector<int> broadcasted_pool_shape((this->rank - _pool_shape.size()), 1);
	broadcasted_pool_shape.insert(broadcasted_pool_shape.end(), _pool_shape.begin(), _pool_shape.end());

	std::vector<int> pool_strides = (_strides.empty()) ? broadcasted_pool_shape : _strides;

	if (pool_strides.size() != this->rank)
	{
		throw std::invalid_argument("[Tensor] Average Pooling failed: stride size mismatch with Tensor's rank.");
	}

	if (!Utils::IsAllPositive(pool_strides))
	{
		throw std::invalid_argument("[Tensor] Average Pooling failed: stride values must be positive.");
	}

	auto pool_volume = Utils::ShapeToVolume(broadcasted_pool_shape);

	auto feature_shape = Utils::ConvolvedFeatureShape(this->shape, broadcasted_pool_shape, pool_strides);
	int feature_volume = Utils::ShapeToVolume(feature_shape);

	std::vector<double> feature_data(feature_volume, 0.0);

	for (int i = 0; i < feature_volume; i++)
	{
		double avg_value = 0;
		auto feature_tensor_index = Utils::TensorIndex(feature_shape, i);

		for (int j = 0; j < pool_volume; j++)
		{
			auto pool_index = Utils::TensorIndex(broadcasted_pool_shape, j);
			auto main_tensor_index = Utils::ScaleNShift(feature_tensor_index, pool_strides, pool_index);

			int flat_index = Utils::FlatIndex(this->shape, main_tensor_index) + this->start_point;
			avg_value += (*this->data)[flat_index];
		}

		feature_data[i] = avg_value / pool_volume;
	}

	return Tensor(feature_shape, feature_data);
}

// ========================================
// Tensor Step-Function Method(s)
// ========================================
Tensor Tensor::Sign(const bool& _heaviside) const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Tensor] Step Function failed: cannot perform sign function on empty Tensor.");
	}

	std::vector<double> sign_data(this->volume, 0.0);

	for (int i = this->start_point, j = 0; i < this->end_point; i++, j++)
	{
		if (std::abs((*this->data)[i]) < std::numeric_limits<double>::epsilon() * EPSILON_SCALE)
		{
			sign_data[j] = 0.0;
		}
		else
		{
			sign_data[j] = ((*this->data)[i] > 0.0) ? 1.0 : -1.0;
		}
		sign_data[j] = (_heaviside && sign_data[j] == 0.0) ? 1.0 : sign_data[j];
	}

	return Tensor(this->shape, sign_data);
}

// ========================================
// Tensor Statistical Method(s)
// ========================================
Tensor Tensor::ReduceSum(const int& _axis) const
{
	if (this->rank == 0)
	{
		throw std::runtime_error("[Tensor] Reduce Sum failed: invalid operation on scalar or empty Tensor.");
	}

	if (_axis < 0 || _axis >= this->rank)
	{
		throw std::out_of_range("[Tensor] Reduce Sum failed: axis out of bounds.");
	}

	std::vector<int> reduced_shape = this->shape;
	reduced_shape.erase(reduced_shape.begin() + _axis);

	Tensor reduced_sum(reduced_shape, 0.0);

	for (int i = 0; i < this->shape[_axis]; i++)
	{
		Tensor temp = this->Slice(_axis, i);
		reduced_sum += temp;
	}

	return reduced_sum;
}

Tensor Tensor::ReduceMean(const int& _axis) const
{
	if (this->rank == 0)
	{
		throw std::runtime_error("[Tensor] Reduce Mean failed: invalid operation on scalar or empty Tensor.");
	}

	if (_axis < 0 || _axis >= this->rank)
	{
		throw std::out_of_range("[Tensor] Reduce Mean failed: axis out of bounds.");
	}

	Tensor reduced_sum = this->ReduceSum(_axis);
	double size = static_cast<double>(this->shape[_axis]);

	return (reduced_sum / size);
}

Tensor Tensor::ReduceVar(const int& _axis, const bool& _inference) const
{
	if (this->rank == 0)
	{
		throw std::runtime_error("[Tensor] Reduce Variance failed: invalid operation on scalar or empty Tensor.");
	}

	if (_axis < 0 || _axis >= this->rank)
	{
		throw std::out_of_range("[Tensor] Reduce Variance failed: axis out of bounds.");
	}

	std::vector<int> reduced_shape = this->shape;
	reduced_shape.erase(reduced_shape.begin() + _axis);

	Tensor reduced_mean = this->ReduceMean(_axis);
	Tensor reduced_var(reduced_shape, 0.0);

	for (int i = 0; i < this->shape[_axis]; i++)
	{
		Tensor temp = this->Slice(_axis, i);
		Tensor diff = temp - reduced_mean;

		reduced_var += (diff * diff);
	}

	double size = static_cast<double>(this->shape[_axis]);
	size -= (_inference && (size > 1)) ? 1 : 0;

	return reduced_var / size;
}

Tensor Tensor::ReduceMax(const int& _axis) const
{
	if (this->rank == 0)
	{
		throw std::runtime_error("[Tensor] Reduce Max failed: invalid operation on scalar or empty Tensor.");
	}

	if (_axis < 0 || _axis >= this->rank)
	{
		throw std::out_of_range("[Tensor] Reduce Max failed: axis out of bounds.");
	}

	std::vector<int> reduced_shape = this->shape;
	reduced_shape.erase(reduced_shape.begin() + _axis);

	Tensor reduced_max = this->Slice(_axis, 0);
	for (int i = 1; i < this->shape[_axis]; i++)
	{
		Tensor temp = this->Slice(_axis, i);
		for (int j = reduced_max.start_point, k = temp.start_point; j < reduced_max.end_point && k < temp.end_point; j++, k++)
		{
			(*reduced_max.data)[j] = std::max((*reduced_max.data)[j], (*temp.data)[k]);
		}
	}

	return reduced_max;
}

Tensor Tensor::ReduceMin(const int& _axis) const
{
	if (this->rank == 0)
	{
		throw std::runtime_error("[Tensor] Reduce Min failed: invalid operation on scalar or empty Tensor.");
	}

	if (_axis < 0 || _axis >= this->rank)
	{
		throw std::out_of_range("[Tensor] Reduce Min failed: axis out of bounds.");
	}

	std::vector<int> reduced_shape = this->shape;
	reduced_shape.erase(reduced_shape.begin() + _axis);

	Tensor reduced_min = this->Slice(_axis, 0);
	for (int i = 1; i < this->shape[_axis]; i++)
	{
		Tensor temp = this->Slice(_axis, i);
		for (int j = reduced_min.start_point, k = temp.start_point; j < reduced_min.end_point && k < temp.end_point; j++, k++)
		{
			(*reduced_min.data)[j] = std::min((*reduced_min.data)[j], (*temp.data)[k]);
		}
	}

	return reduced_min;
}

double Tensor::Sum() const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Tensor] Sum failed: empty Tensor.");
	}

	double sum = std::accumulate(this->data->begin() + this->start_point, this->data->begin() + this->end_point, 0.0);

	return sum;
}

double Tensor::Mean() const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Tensor] Mean failed: empty Tensor.");
	}
	
	double mean = this->Sum() / this->volume;

	return mean;
}

double Tensor::Var(const bool& _inference) const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Tensor] Variance failed: empty Tensor.");
	}
	
	double mean = this->Mean();
	double var = 0.0;

	for (int i = this->start_point; i < this->end_point; i++)
	{
		double diff = (*this->data)[i] - mean;
		var += (diff * diff);
	}

	int size = (_inference && (this->volume > 1)) ? (this->volume - 1) : this->volume;
	var /= size;

	return var;
}

double Tensor::Max() const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Tensor] Max failed: empty Tensor.");
	}

	double max_val = (*this->data)[this->start_point];

	for (int i = this->start_point; i < this->end_point; i++)
	{
		max_val = std::max(max_val, (*this->data)[i]);
	}

	return max_val;
}

double Tensor::Min() const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Tensor] Min failed: empty Tensor.");
	}

	double min_val = (*this->data)[this->start_point];

	for (int i = this->start_point; i < this->end_point; i++)
	{
		min_val = std::min(min_val, (*this->data)[i]);
	}

	return min_val;
}

// ========================================
// Tensor General Mathematical Method(s)
// ========================================
Tensor Tensor::MathOps(const Math::BaseOperation& _math_func) const
{
	return _math_func.f(*this);
}

// ========================================
// Tensor Activation Method(s)
// ========================================
Tensor Tensor::Activate(const Activation::BaseActivation& _activation_func) const
{
	return _activation_func.f(*this);
}

Tensor Tensor::ActivateDerivative(const Activation::BaseActivation& _activation_func) const
{
	return _activation_func.df(*this);
}

// ========================================
// Tensor Utility Method(s)
// ========================================
int Tensor::Rank() const
{
	return this->rank;
}

int Tensor::Volume() const
{
	return this->volume;
}

std::vector<int> Tensor::Shape() const
{
	return this->shape;
}

bool Tensor::IsEmpty() const
{
	return (this->volume == 0);
}

bool Tensor::IsScalar() const
{
	return (this->shape.empty() && this->volume == 1);
}

void Tensor::Clear()
{
	if (this->IsEmpty())
	{
		return;
	}

	this->UniqueData();

	this->rank = 0;
	this->volume = 0;

	this->data->clear();
	this->shape.clear();
	this->strides.clear();

	this->start_point = 0;
	this->end_point = 0;
}

// ========================================
// Tensor Debug Printer
// ========================================
void Tensor::Print(const int& _depth) const
{
	if (this->IsEmpty())
	{
		std::cout << "[]";
		return;
	}

	if (this->IsScalar())
	{
		std::cout << (*this->data)[this->start_point];
		return;
	}

	int outer = this->shape[0];

	std::cout << "\n";
	for (int d = 0; d < _depth; ++d)
	{
		std::cout << " ";
	}
	std::cout << "[";

	for (int i = 0; i < outer; ++i)
	{
		if (i > 0)
		{
			std::cout << ",";
			if (this->rank > 1)
			{
				std::cout << "\n";
			}
		}

		if (this->rank > 1)
		{
			for (int d = 0; d < (_depth + 1); ++d)
			{
				std::cout << " ";
			}
		}

		(*this)[i].Print(_depth + 1);
	}

	if (this->rank > 1)
	{
		std::cout << "\n";
		for (int d = 0; d < _depth; ++d)
		{
			std::cout << " ";
		}
	}
	std::cout << "]";
}

// ========================================
// Tensor Get Method(s)
// ========================================
double Tensor::ToScalar() const
{
	if (!this->IsScalar())
	{
		throw std::runtime_error("[Tensor] Tensor to Scalar failed: Tensor's rank is > 0 (not a scalar).");
	}
	return (*this->data)[this->start_point];
}

std::vector<double> Tensor::ToVector() const
{
	if (this->rank != 1)
	{
		throw std::runtime_error("[Tensor] Tensor to Vector failed: Tensor's rank is not 1 (not a vector).");
	}

	std::vector<double> vec;
	vec.reserve(this->volume);

	for (int i = this->start_point; i < this->end_point; i++)
	{
		vec.push_back((*this->data)[i]);
	}

	return vec;
}

std::vector<std::vector<double>> Tensor::ToMatrix() const
{
	if (this->rank != 2)
	{
		throw std::runtime_error("[Tensor] Tensor to Matrix failed: Tensor's rank is not 2 (not a matrix).");
	}
	
	int rows = this->shape[0];
	int cols = this->shape[1];

	std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int flat_idx = this->start_point + (i * this->strides[0]) + j;
			matrix[i][j] = (*this->data)[flat_idx];
		}
	}

	return matrix;
}

// ========================================
// Tensor Constructing Method(s)
// ========================================
Tensor Tensor::IdentityMatrix(const int& _rows)
{
	if (_rows <= 0)
	{
		throw std::invalid_argument("[Tensor] Identity Matrix Generate failed: matrix size cannot be <= 0.");
	}

	if (Utils::IsVolumeOverflow({ _rows, _rows }))
	{
		throw std::invalid_argument("[Tensor] Identity Matrix Generate failed: shape too large, potential overflow.");
	}

	int volume = (_rows * _rows);
	std::vector<double> data(volume, 0.0);

	for (size_t i = 0; i < volume; i += (_rows + 1))
	{
		data[i] = 1.0;
	}

	return Tensor({ _rows, _rows }, data);
}
