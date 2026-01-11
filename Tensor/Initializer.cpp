#include "Initializer.h"

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

Tensor Initializer::Identity(const std::pair<int, int>& _matrix_axes, const double& scale) const
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

    Tensor I = Tensor::IdentityMatrix(this->shape[axis_1]);

    std::vector<int> expanded_shape(this->rank, 1);
    expanded_shape[axis_1] = this->shape[axis_1];
    expanded_shape[axis_2] = this->shape[axis_2];

    Tensor result = I.Reshape(expanded_shape).Broadcast(this->shape);

    return result;
}

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

Tensor Initializer::Orthogonal(const double& gain) const
{
    // Implement Later.
    return Tensor();
}
