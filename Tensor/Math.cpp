#include "Math.h"
#include "Tensor.h"


// ========================================
// Elementary Methods
// ========================================
Tensor Math::Abs(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        data.push_back(std::abs(value));
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Ceil(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        data.push_back(std::ceil(value));
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Clip(const Tensor& _tensor, const double& _min_value, const double& _max_value)
{
    if (!std::isfinite(_min_value))
    {
        throw std::invalid_argument("[Math] Clip Function failed: min_value must be a finite number.");
    }

    if (!std::isfinite(_max_value))
    {
        throw std::invalid_argument("[Math] Clip Function failed: max_value must be a finite number.");
    }

    if (_min_value > _max_value)
    {
        throw std::invalid_argument("[Math] Clip Function failed: min_value cannot be greater than max_value.");
    }

    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        double result = std::clamp(value, _min_value, _max_value);
        data.push_back(result);
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Exp(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        if (value > Math::EXP_BASE_LIMIT)
        {
            throw std::invalid_argument("[Math] Exponent Function failed: detected large value, " + std::to_string(value) + " - may cause overflow.");
        }

        data.push_back(std::exp(value));
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Floor(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        data.push_back(std::floor(value));
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Log(const Tensor& _tensor, const double& _base)
{
    if (_base <= 0.0 || _base < std::numeric_limits<double>::epsilon() * Math::EPSILON_SCALE)
    {
        throw std::domain_error("[Math] Logarithm Function failed: base cannot be ~ zero or negative.");
    }

    if (std::abs(_base - 1.0) < std::numeric_limits<double>::epsilon() * Math::EPSILON_SCALE)
    {
        throw std::domain_error("[Math] Logarithm Function failed: base cannot be ~ 1.");
    }

    double _log_base = std::log(_base);

    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        if (value <= 0.0 || value < std::numeric_limits<double>::epsilon() * Math::EPSILON_SCALE)
        {
            throw std::invalid_argument("[Math] Logarithm Function failed: detected non-positive value, " + std::to_string(value) + " - logarithm is undefined.");
        }

        double result = std::log(value) / _log_base;
        data.push_back(result);
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Mod(const Tensor& _tensor, const double& _mod_value)
{
    if (std::abs(_mod_value) < std::numeric_limits<double>::epsilon() * Math::EPSILON_SCALE)
    {
        throw std::domain_error("[Math] Modulus Function failed: modulus value cannot be 0 or (~0).");
    }

    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        data.push_back(std::fmod(value, _mod_value));
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Power(const Tensor& _tensor, const double& _exponent)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        if (value < 0.0 && std::fmod(_exponent, 1.0) != 0.0)
        {
            throw std::domain_error("[Math] Power Function failed: negative base detected with non-integer exponent -results non-real number.");
        }

        data.push_back(std::pow(value, _exponent));
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Round(const Tensor& _tensor, const int& _decimal_place)
{
    if (_decimal_place < 0)
    {
        throw std::invalid_argument("[Math] Round Function failed: decimal_place cannot be negative.");
    }

    double power_of_10 = std::pow(10.0, _decimal_place);

    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        double result = std::round(value * power_of_10) / power_of_10;
        data.push_back(result);
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Sqrt(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        if (value < 0.0)
        {
            throw std::domain_error("[Math] Sqrt Function failed: negative value found in input Tensor, " + std::to_string(value));
        }

        data.push_back(std::sqrt(value));
    }

    return Tensor(_tensor.Shape(), data);
}

// ========================================
// Trigonometric Methods
// ========================================
Tensor Math::Sin(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        data.push_back(std::sin(value));
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Cos(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        data.push_back(std::cos(value));
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Tan(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        double cos_value = std::cos(value);

        if (std::abs(cos_value) < std::numeric_limits<double>::epsilon() * Math::EPSILON_SCALE)
        {
            throw std::domain_error("[Math] Tangent Function failed: tangent is undefined near odd multiples of `pi`/2.");
        }

        data.push_back(std::tan(value));
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Csc(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        double sin_value = std::sin(value);

        if (std::abs(sin_value) < std::numeric_limits<double>::epsilon() * Math::EPSILON_SCALE)
        {
            throw std::domain_error("[Math] Cosecant Function failed: cosecant is undefined near multiples of `pi`.");
        }

        data.push_back(1.0 / sin_value);
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Sec(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        double cos_value = std::cos(value);

        if (std::abs(cos_value) < std::numeric_limits<double>::epsilon() * Math::EPSILON_SCALE)
        {
            throw std::domain_error("[Math] Secant Function failed: secant is undefined near odd multiples of `pi`/2.");
        }

        data.push_back(1.0 / cos_value);
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Cot(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        double sin_value = std::sin(value);

        if (std::abs(sin_value) < std::numeric_limits<double>::epsilon() * Math::EPSILON_SCALE)
        {
            throw std::domain_error("[Math] Cotangent Function failed: cotangent is undefined near multiples of `pi`.");
        }

        data.push_back(std::cos(value) / sin_value);
    }

    return Tensor(_tensor.Shape(), data);
}

// ========================================
// Inverse Trigonometric Methods
// ========================================
Tensor Math::Asin(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        if (value < -1.0 || value > 1.0)
        {
            throw std::domain_error("[Math] Arc-sine Function failed: arcsine is only defined for values in [-1, 1].");
        }

        data.push_back(std::asin(value));
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Acos(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        if (value < -1.0 || value > 1.0)
        {
            throw std::domain_error("[Math] Arc-cosine Function failed: arccosine is only defined for values in [-1, 1].");
        }

        data.push_back(std::acos(value));
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Atan(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        data.push_back(std::atan(value));
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Acsc(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        if (value > -1.0 && value < 1.0)
        {
            throw std::domain_error("[Math] Arc-cosecant Function failed: arccosecant is not defined for values in (-1, 1).");
        }

        data.push_back(std::asin(1.0 / value));
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Asec(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        if (value > -1.0 && value < 1.0)
        {
            throw std::domain_error("[Math] Arc-secant Function failed: arcsecant is not defined for values in (-1, 1).");
        }

        data.push_back(std::acos(1.0 / value));
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Acot(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    double acot_value = 0.0;

    for (const double& value : _tensor)
    {
        if (std::abs(value) < std::numeric_limits<double>::epsilon() * Math::EPSILON_SCALE)
        {
            acot_value = std::numbers::pi / 2.0;
        }
        else
        {
            acot_value = std::atan(1.0 / value);
            acot_value += (value < 0.0) ? std::numbers::pi : 0.0;
        }

        data.push_back(acot_value);
    }

    return Tensor(_tensor.Shape(), data);
}

// ========================================
// Hyperbolic Methods
// ========================================
Tensor Math::Sinh(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        data.push_back(std::sinh(value));
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Cosh(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        data.push_back(std::cosh(value));
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Tanh(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        data.push_back(std::tanh(value));
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Csch(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        double sinh_value = std::sinh(value);

        if (std::abs(sinh_value) < std::numeric_limits<double>::epsilon() * Math::EPSILON_SCALE)
        {
            throw std::domain_error("[Math] Hyperbolic Cosecant Function failed: hyperbolic cosecant is undefined at ~ zero.");
        }

        data.push_back(1.0 / sinh_value);
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Sech(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        data.push_back(1.0 / std::cosh(value));
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Coth(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        double sinh_value = std::sinh(value);

        if (std::abs(sinh_value) < std::numeric_limits<double>::epsilon() * Math::EPSILON_SCALE)
        {
            throw std::domain_error("[Math] Hyperbolic Cotangent Function failed: hyperbolic cotangent is undefined at ~ zero.");
        }

        data.push_back(std::cosh(value) / sinh_value);
    }

    return Tensor(_tensor.Shape(), data);
}

// ========================================
// Inverse Hyperbolic Methods
// ========================================
Tensor Math::Asinh(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        data.push_back(std::asinh(value));
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Acosh(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        if (value < 1.0)
        {
            throw std::domain_error("[Math] Inverse Hyperbolic Cosine Function failed: inverse hyperbolic cosine is only defined for values >= 1.");
        }

        data.push_back(std::acosh(value));
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Atanh(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        if (value <= -1.0 || value >= 1.0)
        {
            throw std::domain_error("[Math] Inverse Hyperbolic Tangent Function failed: inverse hyperbolic tangent is only defined for values in (-1, 1).");
        }

        data.push_back(std::atanh(value));
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Acsch(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        if (std::abs(value) < std::numeric_limits<double>::epsilon() * Math::EPSILON_SCALE)
        {
            throw std::domain_error("[Math] Inverse Hyperbolic Cosecant Function failed: inverse hyperbolic cosecant is undefined at zero.");
        }

        data.push_back(std::asinh(1.0 / value));
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Asech(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        if (value <= 0.0 || value > 1.0)
        {
            throw std::domain_error("[Math] Inverse Hyperbolic Secant Function failed: inverse hyperbolic secant is only defined for values in (0, 1].");
        }

        data.push_back(std::acosh(1.0 / value));
    }

    return Tensor(_tensor.Shape(), data);
}

Tensor Math::Acoth(const Tensor& _tensor)
{
    std::vector<double> data;
    data.reserve(_tensor.Volume());

    for (const double& value : _tensor)
    {
        if (value >= -1.0 && value <= 1.0)
        {
            throw std::domain_error("[Math] Inverse Hyperbolic Cotangent Function failed: inverse hyperbolic cotangent is not defined for values in [-1, 1].");
        }

        data.push_back(std::atanh(1.0 / value));
    }

    return Tensor(_tensor.Shape(), data);
}
