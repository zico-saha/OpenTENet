#include "Initializer.h"

// ========================================
// [Private] Generator Setup Method
// ========================================
void Initializer::InitializeGenerator()
{
    if (this->seed.has_value())
    {
        this->generator.seed(this->seed.value());
    }
    else
    {
        std::random_device rd;
        this->generator.seed(rd());
    }
}

// ========================================
// Initializer Constructor(s)
// ========================================
Initializer::Initializer()
    : seed(std::nullopt)
{
    this->rank = 0;
    this->volume = 1;

    this->InitializeGenerator();
}

Initializer::Initializer(const std::vector<int>& _shape, const std::optional<unsigned int>& _seed)
    : shape(_shape), seed(_seed)
{
    if (Utils::IsVolumeOverflow(_shape))
    {
        throw std::overflow_error("[Initializer] Constructing Tensor failed: shape too large, potential overflow.");
    }

    this->rank = static_cast<int>(_shape.size());
    this->volume = Utils::ShapeToVolume(_shape);

    this->InitializeGenerator();
}

// ========================================
// Constant-Initializer Method(s)
// ========================================
Tensor Initializer::Zeros() const
{
    return Tensor(this->shape, 0.0);
}

Tensor Initializer::Ones() const
{
    return Tensor(this->shape, 1.0);
}

Tensor Initializer::Constant(const double& _value) const
{
    return Tensor(this->shape, _value);
}

Tensor Initializer::Identity(const std::pair<int, int>& _matrix_axes, const double& _scale) const
{
    if (this->rank <= 1)
    {
        throw std::runtime_error("[Initializer] Identity Tensor Construct failed: Identity requires at least 2D tensor.");
    }

    int axis_1 = (_matrix_axes.first < 0) ? (_matrix_axes.first + this->rank) : _matrix_axes.first;
    int axis_2 = (_matrix_axes.second < 0) ? (_matrix_axes.second + this->rank) : _matrix_axes.second;

    if (axis_1 < 0 || axis_1 >= this->rank)
    {
        throw std::invalid_argument("[Initializer] Identity Tensor Construct failed: first matrix-axis- " + std::to_string(_matrix_axes.first) + " is out-of-bound with Tensor rank.");
    }

    if (axis_2 < 0 || axis_2 >= this->rank)
    {
        throw std::invalid_argument("[Initializer] Identity Tensor Construct failed: second matrix-axis- " + std::to_string(_matrix_axes.second) + " is out-of-bound with Tensor rank.");
    }

    if (this->shape[axis_1] != this->shape[axis_2])
    {
        throw std::runtime_error("[Initializer] Identity Tensor Construct failed: identity axes must have equal size.");
    }

    LinAlg::Matrix identity_matrix = LinAlg::Matrix::Identity(this->shape[axis_1], _scale);
    Tensor identity_tensor(identity_matrix);

    std::vector<int> expanded_shape(this->rank, 1);
    expanded_shape[axis_1] = this->shape[axis_1];
    expanded_shape[axis_2] = this->shape[axis_2];

    Tensor result = identity_tensor.Reshape(expanded_shape).Broadcast(this->shape);

    return result;
}

// ========================================
// RandomDist-Initializer Method(s)
// ========================================
Tensor Initializer::RandomNormal(const double& _mean, const double& _std_dev) const
{
    std::vector<double> data(this->volume);
    std::normal_distribution<double> dist(_mean, _std_dev);

    for (int i = 0; i < this->volume; i++)
    {
        data[i] = dist(this->generator);
    }

    return Tensor(this->shape, data);
}

Tensor Initializer::RandomUniform(const double& _min_val, const double& _max_val) const
{
    std::vector<double> data(this->volume);
    std::uniform_real_distribution<double> dist(_min_val, _max_val);

    for (int i = 0; i < this->volume; i++)
    {
        data[i] = dist(this->generator);
    }

    return Tensor(this->shape, data);
}

Tensor Initializer::TruncatedNormal(const double& _mean, const double& _std_dev, const double& _truncate_std_dev_scale) const
{
    std::vector<double> data(this->volume);
    std::normal_distribution<double> dist(_mean, _std_dev);

    double lower_bound = _mean - (_truncate_std_dev_scale * _std_dev);
    double upper_bound = _mean + (_truncate_std_dev_scale * _std_dev);

    for (int i = 0; i < this->volume; i++)
    {
        double value;
        do
        {
            value = dist(generator);
        } while (value < lower_bound || value > upper_bound);

        data[i] = value;
    }

    return Tensor(this->shape, data);
}

// ========================================
// Special RandomDist-Initializer Method(s)
// ========================================
Tensor Initializer::GlorotNormal(const int& _fan_in, const int& _fan_out) const
{
    double std_dev = std::sqrt(2.0 / (_fan_in + _fan_out));
    return this->RandomNormal(0.0, std_dev);
}

Tensor Initializer::GlorotUniform(const int& _fan_in, const int& _fan_out) const
{
    double limit = std::sqrt(6.0 / (_fan_in + _fan_out));
    return this->RandomUniform(-limit, limit);
}

Tensor Initializer::HeNormal(const int& _fan_in) const
{
    double std_dev = std::sqrt(2.0 / _fan_in);
    return this->RandomNormal(0.0, std_dev);
}

Tensor Initializer::HeUniform(const int& _fan_in) const
{
    double limit = std::sqrt(6.0 / _fan_in);
    return this->RandomUniform(-limit, limit);
}

Tensor Initializer::LecunNormal(const int& _fan_in) const
{
    double std_dev = std::sqrt(1.0 / _fan_in);
    return this->RandomNormal(0.0, std_dev);
}

Tensor Initializer::LecunUniform(const int& _fan_in) const
{
    double limit = std::sqrt(3.0 / _fan_in);
    return this->RandomUniform(-limit, limit);
}

// ========================================
// Orthogonal-Matrix-Initializer Method(s)
// ========================================
Tensor Initializer::Orthogonal(const std::pair<int, int>& _axes, const double& _gain) const
{
    if (this->shape.size() < 2)
    {
        throw std::invalid_argument("[Initializer] Orthogonal Initialization failed: requires at least 2D Tensor.");
    }

    int axis_1 = (_axes.first < 0) ? (_axes.first + this->rank) : _axes.first;
    int axis_2 = (_axes.second < 0) ? (_axes.second + this->rank) : _axes.second;

    if (axis_1 < 0 || axis_1 >= this->rank)
    {
        throw std::out_of_range("[Initializer] Orthogonal Initialization failed: first axis (_axes.first) is out of bounds with Tensor rank.");
    }

    if (axis_2 < 0 || axis_2 >= this->rank)
    {
        throw std::out_of_range("[Initializer] Orthogonal Initialization failed: second axis (_axes.first) is out of bounds with Tensor rank.");
    }

    int rows = this->shape[axis_1];
    int columns = this->shape[axis_2];

    int matrix_vol = rows * columns;
    int n_matrix = this->volume / matrix_vol;

    std::vector<int> permutation;
    permutation.reserve(this->rank);

    for (int i = 0; i < this->rank; i++)
    {
        if (i != axis_1 && i != axis_2)
        {
            permutation.push_back(i);
        }
    }
    permutation.push_back(axis_1);
    permutation.push_back(axis_2);

    std::vector<int> permuted_shape = Utils::Permute(this->shape, permutation);

    std::vector<double> tensor_data;
    tensor_data.reserve(this->volume);

    for (int i = 0; i < n_matrix; i++)
    {
        LinAlg::Matrix matrix = LinAlg::Matrix::RandomNormal(rows, columns, 0.0, 1.0);
        LinAlg::QRResult qr = matrix.HQRDecomposition(false);

        std::vector<double> sign_values = qr.Q.Diag(true);

        LinAlg::Matrix result = qr.Q * sign_values;
        result = result.MultiplyColumnwise(sign_values);

        std::vector<double> flat_data = result.GetFlatData();
        tensor_data.insert(tensor_data.end(), flat_data.begin(), flat_data.end());
    }

    Tensor param_tensor(permuted_shape, tensor_data);
    param_tensor *= _gain;

    std::vector<int> rev_permutation(this->rank);
    for (int i = 0; i < this->rank; i++)
    {
        rev_permutation[permutation[i]] = i;
    }

    return param_tensor.Transpose(rev_permutation);
}
