#include "Sparsemax.h"
#include "Tensor.h"

Activation::Sparsemax::Sparsemax(int axis)
{
	this->axis = axis;
}

std::vector<double> Activation::Sparsemax::__Sparsemax(const std::vector<double>& nums) const
{
    int n = static_cast<int>(nums.size());

    std::vector<double> nums_sorted = nums;
    std::sort(nums_sorted.begin(), nums_sorted.end(), std::greater<double>());

    std::vector<double> cumsum(n);
    std::partial_sum(nums_sorted.begin(), nums_sorted.end(), cumsum.begin());

    int k_star = n;
    for (int k = 1; k <= n; k++)
    {
        double tau_k = (cumsum[k - 1] - 1.0) / k;
        if ((nums_sorted[k - 1] - tau_k) <= 0.0)
        {
            k_star = k - 1;
            break;
        }
    }

    double tau = (cumsum[k_star - 1] - 1.0) / k_star;

    std::vector<double> sparse_prob(n);
    for (int i = 0; i < n; i++)
    {
        sparse_prob[i] = std::max((nums[i] - tau), 0.0);
    }

    return sparse_prob;
}

Tensor Activation::Sparsemax::f(const Tensor& tensor) const
{
    if (tensor.IsEmpty())
    {
        throw std::runtime_error("[Activation] Sparsemax failed: empty Tensor.");
    }

    if (tensor.IsScalar())
    {
        throw std::invalid_argument("[Activation] Sparsemax failed: cannot apply Sparsemax to scalar. Sparsemax requires at least 2 elements.");
    }

    int actual_axis = (this->axis < 0) ? (tensor.Rank() + this->axis) : this->axis;

    if (actual_axis < 0 || actual_axis >= tensor.Rank())
    {
        throw std::out_of_range("[Activation] Sparsemax failed: axis out of range.");
    }

    std::vector<int> permutation(tensor.Rank());
    std::iota(permutation.begin(), permutation.end(), 0);
    permutation.erase(permutation.begin() + actual_axis);
    permutation.push_back(actual_axis);

    Tensor transposed_tensor = tensor.Transpose(permutation);

    int len = tensor.Shape()[actual_axis];
    std::vector<double> nums;
    nums.reserve(len);

    std::vector<double> activated_data;
    activated_data.reserve(tensor.Volume());

    for (const double& num : transposed_tensor)
    {
        nums.push_back(num);

        if (nums.size() == len)
        {
            auto result = this->__Sparsemax(nums);
            activated_data.insert(activated_data.end(), result.begin(), result.end());
            nums.clear();
        }
    }

    Tensor transposed_result(transposed_tensor.Shape(), activated_data);

    permutation.insert((permutation.begin() + actual_axis), (tensor.Rank() - 1));
    permutation.pop_back();
    for (int i = actual_axis + 1; i < permutation.size(); i++)
    {
        permutation[i]--;
    }

    return transposed_result.Transpose(permutation);
}

Tensor Activation::Sparsemax::df(const Tensor& tensor) const
{
    if (tensor.IsEmpty())
    {
        throw std::runtime_error("[Activation] Sparsemax failed: empty Tensor.");
    }

    if (tensor.IsScalar())
    {
        throw std::invalid_argument("[Activation] Sparsemax failed: cannot apply Sparsemax to scalar. Sparsemax requires at least 2 elements.");
    }

    int actual_axis = (this->axis < 0) ? (tensor.Rank() + this->axis) : this->axis;

    if (actual_axis < 0 || actual_axis >= tensor.Rank())
    {
        throw std::out_of_range("[Activation] Sparsemax failed: axis out of range.");
    }

    std::vector<int> permutation(tensor.Rank());
    std::iota(permutation.begin(), permutation.end(), 0);
    permutation.erase(permutation.begin() + actual_axis);
    permutation.push_back(actual_axis);

    Tensor transposed_tensor = tensor.Transpose(permutation);

    int len = tensor.Shape()[actual_axis];
    std::vector<double> nums;
    nums.reserve(len);

    std::vector<double> jacobian_data;

    for (const double& num : transposed_tensor)
    {
        nums.push_back(num);

        if (nums.size() == len)
        {
            auto sparse_output = this->__Sparsemax(nums);

            std::vector<bool> support(len, false);
            int support_size = 0;

            for (int i = 0; i < len; i++)
            {
                if (sparse_output[i] > Activation::Sparsemax::EPSILON)
                {
                    support[i] = true;
                    support_size++;
                }
            }

            double inv_support_size = 1.0 / support_size;

            for (int i = 0; i < len; i++)
            {
                for (int j = 0; j < len; j++)
                {
                    double jacobian_ij = 0;
                    if (support[i] && support[j])
                    {
                        jacobian_ij = (i == j) ? (1.0 - inv_support_size) : (-inv_support_size);
                    }
                    jacobian_data.push_back(jacobian_ij);
                }
            }

            nums.clear();
        }
    }

    std::vector<int> jacobian_shape = transposed_tensor.Shape();
    jacobian_shape.push_back(len);

    Tensor jacobian_tensor(jacobian_shape, jacobian_data);

    std::vector<int> inv_permutation(jacobian_tensor.Rank());
    std::iota(inv_permutation.begin(), inv_permutation.end(), 0);
    std::rotate(inv_permutation.begin() + actual_axis, inv_permutation.end() - 2, inv_permutation.end());

    return jacobian_tensor.Transpose(inv_permutation);
}
