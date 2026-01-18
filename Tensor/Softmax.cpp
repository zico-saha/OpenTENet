#include "Softmax.h"
#include "Math.h"
#include "Matrix.h"
#include "Tensor.h"


Activation::Softmax::Softmax(int axis)
{
    this->axis = axis;
}

Tensor Activation::Softmax::f(const Tensor& tensor) const
{
    if (tensor.IsEmpty())
    {
        throw std::runtime_error("[Activation] Softmax failed: empty Tensor.");
    }

    if (tensor.IsScalar())
    {
        throw std::invalid_argument("[Activation] Softmax failed: cannot apply Softmax to scalar. Softmax requires at least 2 elements.");
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

    return exp_vals / sum_exp.ExpandRank(actual_axis);
}

Tensor Activation::Softmax::df(const Tensor& tensor) const
{
    Tensor result = this->f(tensor);

    int actual_axis = (this->axis < 0) ? (tensor.Rank() + this->axis) : this->axis;
    int size = result.Shape()[actual_axis];

    std::vector<int> permutation_1(result.Rank());
    std::iota(permutation_1.begin(), permutation_1.end(), 0);

    permutation_1.erase(permutation_1.begin() + actual_axis);
    permutation_1.push_back(actual_axis);

    result = result.Transpose(permutation_1);

    Tensor s1 = result.ExpandRank(result.Rank());
    Tensor s2 = result.ExpandRank(result.Rank() - 1);

    Tensor jacobian = Tensor::MatMul(s1, s2);

    Tensor identity_matrix(LinAlg::Matrix::Identity(size));
    Tensor identity_tensor = identity_matrix.Broadcast(jacobian.Shape());

    jacobian = (s1 * identity_tensor) - jacobian;

    std::vector<int> permutation_2(jacobian.Rank());
    std::iota(permutation_2.begin(), permutation_2.end(), 0);
    std::rotate(permutation_2.begin() + actual_axis, permutation_2.end() - 2, permutation_2.end());

    return jacobian.Transpose(permutation_2);
}
