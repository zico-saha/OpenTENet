#include "TensorSlice.h"
#include "Tensor.h"

// ========================================
// [PRIVATE] TensorSlice -> Parent Tensor Method
// ========================================
Tensor TensorSlice::_Tensor() const
{
    return this->root_parent->GetSliceChain(this->index_chain);
}

TensorSlice::SliceInfo TensorSlice::_GetDirectAccess() const
{
    TensorSlice::SliceInfo info;
    info.data = root_parent->data;

    int current_offset = root_parent->start_point;
    std::vector<int> current_shape = root_parent->shape;
    std::vector<int> current_strides = root_parent->strides;

    for (int idx : index_chain)
    {
        current_offset += idx * current_strides[0];
        current_shape.erase(current_shape.begin());
        current_strides.erase(current_strides.begin());
    }

    int volume = 1;
    for (int dim : current_shape)
    {
        volume *= dim;
    }

    info.start_offset = current_offset;
    info.end_offset = current_offset + volume;

    return info;
}

// ========================================
// TensorSlice -> Tensor Conversion Operator
// ========================================
TensorSlice::operator Tensor() const
{
    return this->root_parent->GetSliceChain(this->index_chain);
}

// ========================================
// TensorSlice Assignment Operator
// ========================================
TensorSlice& TensorSlice::operator=(const Tensor& _tensor)
{
    this->root_parent->SetSliceChain(this->index_chain, _tensor);
    return *this;
}

// ========================================
// TensorSlice Indexing Operator(s)
// ========================================
TensorSlice TensorSlice::operator[](const int& index)
{
    std::vector<int> new_chain = this->index_chain;
    new_chain.push_back(index);
    return TensorSlice(this->root_parent, new_chain);
}

Tensor TensorSlice::operator[](const int& index) const
{
    std::vector<int> new_chain = this->index_chain;
    new_chain.push_back(index);
    return this->root_parent->GetSliceChain(new_chain);
}

// ========================================
// Pass-through Tensor Iterator(s)
// ========================================
TensorSlice::iterator TensorSlice::begin()
{
    auto info = this->_GetDirectAccess();
    return info.data->begin() + info.start_offset;
}

TensorSlice::iterator TensorSlice::end()
{
    auto info = this->_GetDirectAccess();
    return info.data->begin() + info.end_offset;
}

TensorSlice::const_iterator TensorSlice::begin() const
{
    auto info = this->_GetDirectAccess();
    return info.data->begin() + info.start_offset;
}

TensorSlice::const_iterator TensorSlice::end() const
{
    auto info = this->_GetDirectAccess();
    return info.data->begin() + info.end_offset;
}

// ========================================
// Pass-through Tensor Arithmetic Operator(s)
// ========================================
Tensor TensorSlice::operator+(const double& _value) const
{
    return this->_Tensor() + _value;
}

Tensor TensorSlice::operator-(const double& _value) const
{
    return this->_Tensor() - _value;
}

Tensor TensorSlice::operator*(const double& _value) const
{
    return this->_Tensor() * _value;
}

Tensor TensorSlice::operator/(const double& _value) const
{
    return this->_Tensor() / _value;
}

Tensor TensorSlice::operator+(const Tensor& _tensor) const
{
    return this->_Tensor() + _tensor;
}

Tensor TensorSlice::operator-(const Tensor& _tensor) const
{
    return this->_Tensor() - _tensor;
}

Tensor TensorSlice::operator*(const Tensor& _tensor) const
{
    return this->_Tensor() * _tensor;
}

Tensor TensorSlice::operator/(const Tensor& _tensor) const
{
    return this->_Tensor() / _tensor;
}

TensorSlice& TensorSlice::operator+=(const double& _value)
{
    Tensor temp = this->_Tensor();
    temp += _value;

    this->root_parent->SetSliceChain(this->index_chain, temp);

    return *this;
}

TensorSlice& TensorSlice::operator-=(const double& _value)
{
    Tensor temp = this->_Tensor();
    temp -= _value;

    this->root_parent->SetSliceChain(this->index_chain, temp);

    return *this;
}

TensorSlice& TensorSlice::operator*=(const double& _value)
{
    Tensor temp = this->_Tensor();
    temp *= _value;

    this->root_parent->SetSliceChain(this->index_chain, temp);

    return *this;
}

TensorSlice& TensorSlice::operator/=(const double& _value)
{
    Tensor temp = this->_Tensor();
    temp /= _value;

    this->root_parent->SetSliceChain(this->index_chain, temp);

    return *this;
}

TensorSlice& TensorSlice::operator+=(const Tensor& _tensor)
{
    Tensor temp = this->_Tensor();
    temp = temp + _tensor;

    this->root_parent->SetSliceChain(this->index_chain, temp);

    return *this;
}

TensorSlice& TensorSlice::operator-=(const Tensor& _tensor)
{
    Tensor temp = this->_Tensor();
    temp = temp - _tensor;

    this->root_parent->SetSliceChain(this->index_chain, temp);

    return *this;
}

TensorSlice& TensorSlice::operator*=(const Tensor& _tensor)
{
    Tensor temp = this->_Tensor();
    temp = temp * _tensor;

    this->root_parent->SetSliceChain(this->index_chain, temp);

    return *this;
}

TensorSlice& TensorSlice::operator/=(const Tensor& _tensor)
{
    Tensor temp = this->_Tensor();
    temp = temp / _tensor;

    this->root_parent->SetSliceChain(this->index_chain, temp);

    return *this;
}

// ========================================
// Pass-through Tensor Reshaping Method(s)
// ========================================
Tensor TensorSlice::Reshape(const std::vector<int>& _new_shape) const
{
    return this->_Tensor().Reshape(_new_shape);
}

Tensor TensorSlice::ExpandRank(const int& _axis) const
{
    return this->_Tensor().ExpandRank(_axis);
}

Tensor TensorSlice::Flatten(const int& _axis_from, const int& _axis_upto) const
{
    return this->_Tensor().Flatten(_axis_from, _axis_upto);
}

// ========================================
// Pass-through Tensor Slicing Method(s)
// ========================================
Tensor TensorSlice::Slice(const int& _axis, const int& _index) const
{
    return this->_Tensor().Slice(_axis, _index);
}

Tensor TensorSlice::Slice(const int& _axis, const int& _index_from, const int& _index_upto) const
{
    return this->_Tensor().Slice(_axis, _index_from, _index_upto);
}

// ========================================
// Pass-through Tensor Appending Method(s)
// ========================================
Tensor TensorSlice::Pad(const int& _axis, const int& _pad_before_size, const int& _pad_after_size, const double& _value) const
{
    return this->_Tensor().Pad(_axis, _pad_before_size, _pad_after_size, _value);
}

Tensor TensorSlice::Tile(const std::vector<int>& _repetitions) const
{
    return this->_Tensor().Tile(_repetitions);
}

// ========================================
// Pass-through Tensor Broadcasting Method
// ========================================
Tensor TensorSlice::Broadcast(const std::vector<int>& _shape) const
{
    return this->_Tensor().Broadcast(_shape);
}

// ========================================
// Pass-through Tensor Transpose Method
// ========================================
Tensor TensorSlice::Transpose(const std::vector<int>& _permutation) const
{
    return this->_Tensor().Transpose(_permutation);
}

// ========================================
// Pass-through Tensor Dot Product Method(s)
// ========================================
Tensor TensorSlice::MatMul(const Tensor& _tensor) const
{
    return this->_Tensor().MatMul(_tensor);
}

// ========================================
// Pass-Through Tensor Convolution Method(s)
// ========================================
Tensor TensorSlice::Convolve(const Tensor& _filter, const std::vector<int>& _strides, const std::vector<int>& _padding)
{
    return this->_Tensor().Convolve(_filter, _strides, _padding);
}

// ========================================
// Pass-through Tensor Pooling Method(s)
// ========================================
Tensor TensorSlice::MaxPool(const std::vector<int>& _pool_shape, const std::vector<int>& _strides)
{
    return this->_Tensor().MaxPool(_pool_shape, _strides);
}

Tensor TensorSlice::MinPool(const std::vector<int>& _pool_shape, const std::vector<int>& _strides)
{
    return this->_Tensor().MinPool(_pool_shape, _strides);
}

Tensor TensorSlice::AvgPool(const std::vector<int>& _pool_shape, const std::vector<int>& _strides)
{
    return this->_Tensor().AvgPool(_pool_shape, _strides);
}

// ========================================
// Pass-through Tensor Step-Function Method(s)
// ========================================
Tensor TensorSlice::Sign(const bool& _heaviside) const
{
    return this->_Tensor().Sign(_heaviside);
}

// ========================================
// Pass-through Tensor Statistical Method(s)
// ========================================
Tensor TensorSlice::ReduceSum(const int& _axis) const
{
    return this->_Tensor().ReduceSum(_axis);
}

Tensor TensorSlice::ReduceMean(const int& _axis) const
{
    return this->_Tensor().ReduceMean(_axis);
}

Tensor TensorSlice::ReduceVar(const int& _axis, const bool& _inference) const
{
    return this->_Tensor().ReduceVar(_axis, _inference);
}

Tensor TensorSlice::ReduceMax(const int& _axis) const
{
    return this->_Tensor().ReduceMax(_axis);
}

Tensor TensorSlice::ReduceMin(const int& _axis) const
{
    return this->_Tensor().ReduceMin(_axis);
}

double TensorSlice::Sum() const
{
    return this->_Tensor().Sum();
}

double TensorSlice::Mean() const
{
    return this->_Tensor().Mean();
}

double TensorSlice::Var(const bool& _inference) const
{
    return this->_Tensor().Var(_inference);
}

double TensorSlice::Max() const
{
    return this->_Tensor().Max();
}

double TensorSlice::Min() const
{
    return this->_Tensor().Min();
}

// ========================================
// Tensor General Mathematical Method(s)
// ========================================
Tensor TensorSlice::MathOps(const Math::BaseOperation& _math_func) const
{
    Tensor temp = this->_Tensor();
    return _math_func.f(temp);
}

// ========================================
// Pass-through Tensor Activation Method(s)
// ========================================
Tensor TensorSlice::Activate(const Activation::BaseActivation& _activation_func) const
{
    Tensor temp = this->_Tensor();
    return _activation_func.f(temp);
}

Tensor TensorSlice::ActivateDerivative(const Activation::BaseActivation& _activation_func) const
{
    Tensor temp = this->_Tensor();
    return _activation_func.df(temp);
}

// ========================================
// Pass-through Tensor Utility Method(s)
// ========================================
int TensorSlice::Rank() const
{
    return this->_Tensor().Rank();
}

int TensorSlice::Volume() const
{
    return this->_Tensor().Volume();
}

std::vector<int> TensorSlice::Shape() const
{
    return this->_Tensor().Shape();
}

bool TensorSlice::IsEmpty() const
{
    return this->_Tensor().IsEmpty();
}

bool TensorSlice::IsScalar() const
{
    return this->_Tensor().IsScalar();
}

// ========================================
// Pass-through Tensor Debug Printer
// ========================================
void TensorSlice::Print(const int& _depth) const
{
    this->_Tensor().Print(_depth);
}

// ========================================
// Pass-through Tensor Get Method(s)
// ========================================
double TensorSlice::ToScalar() const
{
    return this->_Tensor().ToScalar();
}

std::vector<double> TensorSlice::ToVector() const
{
    return this->_Tensor().ToVector();
}

std::vector<std::vector<double>> TensorSlice::ToMatrix() const
{
    return this->_Tensor().ToMatrix();
}
