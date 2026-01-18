#include "LogSoftmax.h"
#include "Math.h"
#include "Matrix.h"
#include "Tensor.h"


Activation::LogSoftmax::LogSoftmax(int axis)
{
    this->axis = axis;
}

Tensor Activation::LogSoftmax::f(const Tensor& tensor) const
{
    if (tensor.IsEmpty())
    {
        throw std::runtime_error("[Activation] LogSoftmax failed: empty Tensor.");
    }

    if (tensor.IsScalar())
    {
        throw std::invalid_argument("[Activation] LogSoftmax failed: cannot apply LogSoftmax to scalar. LogSoftmax requires at least 2 elements.");
    }

    int actual_axis = (this->axis < 0) ? (tensor.Rank() + this->axis) : this->axis;

    if (actual_axis < 0 || actual_axis >= tensor.Rank())
    {
        throw std::out_of_range("[Activation] Softmax failed: axis out of range.");
    }

    Tensor max_vals = tensor.ReduceMax(actual_axis);
    Tensor shifted = tensor - max_vals.ExpandRank(actual_axis);
    Tensor exp_vals = Math::Exp(shifted);
    Tensor sum_exp = exp_vals.ReduceSum(actual_axis);

    Tensor log_sum = Math::Log(sum_exp);

    return (shifted - log_sum.ExpandRank(actual_axis));
}

Tensor Activation::LogSoftmax::df(const Tensor& tensor) const
{
    auto func = Activation::Softmax(this->axis);
    Tensor result = func.f(tensor);

    int actual_axis = (this->axis < 0) ? (tensor.Rank() + this->axis) : this->axis;
    int size = result.Shape()[actual_axis];

    result = result.ExpandRank(actual_axis);

    auto broadcast_shape = result.Shape();
    broadcast_shape[actual_axis] = broadcast_shape[actual_axis + 1];

    result = result.Broadcast(broadcast_shape);

    Tensor identity_matrix(LinAlg::Matrix::Identity(size));

    std::vector<int> shape(result.Rank(), 1);
    shape[actual_axis] = size;
    shape[actual_axis + 1] = size;

    identity_matrix = identity_matrix.Reshape(shape);

    Tensor jacobian = identity_matrix - result;

    return jacobian;
}
