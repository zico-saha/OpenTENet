// ========================================
// Elementary Methods
// ========================================
template <MathContainer T>
T Math::Abs(const T& x)
{
    return x.Apply([](double value) {
        return std::abs(value);
        });
}

template <MathContainer T>
T Math::Ceil(const T& x)
{
    return x.Apply([](double value) {
        return std::ceil(value);
        });
}

template <MathContainer T>
T Math::Clip(const T& x, double min_value, double max_value)
{
    if (!std::isfinite(min_value))
    {
        throw std::invalid_argument("[Math] Clip Function failed: min_value must be a finite number.");
    }

    if (!std::isfinite(max_value))
    {
        throw std::invalid_argument("[Math] Clip Function failed: max_value must be a finite number.");
    }

    if (min_value > max_value)
    {
        throw std::invalid_argument("[Math] Clip Function failed: min_value cannot be greater than max_value.");
    }

    return x.Apply([=](double value) {
        return std::clamp(value, min_value, max_value);
        });
}

template <MathContainer T>
T Math::Exp(const T& x)
{
    return x.Apply([](double value) {
        if (value > EXP_BASE_LIMIT)
        {
            throw std::invalid_argument("[Math] Exponent Function failed: detected large value, " + std::to_string(value) + " - may cause overflow.");
        }

        return std::exp(value);
        });
}

template <MathContainer T>
T Math::Floor(const T& x)
{
    return x.Apply([](double value) {
        return std::floor(value);
        });
}

template <MathContainer T>
T Math::Log(const T& x, double base)
{
    if (base <= 0.0 || base < std::numeric_limits<double>::epsilon() * EPSILON_SCALE)
    {
        throw std::domain_error("[Math] Logarithm Function failed: base cannot be ~ zero or negative.");
    }

    if (std::abs(base - 1.0) < std::numeric_limits<double>::epsilon() * EPSILON_SCALE)
    {
        throw std::domain_error("[Math] Logarithm Function failed: base cannot be ~ 1.");
    }

    const double log_base = std::log(base);

    return x.Apply([=](double value) {
        if (value <= 0.0 || value < std::numeric_limits<double>::epsilon() * EPSILON_SCALE)
        {
            throw std::invalid_argument("[Math] Logarithm Function failed: detected non-positive value, " + std::to_string(value) + " - logarithm is undefined.");
        }

        return std::log(value) / log_base;
        });
}

template <MathContainer T>
T Math::Mod(const T& x, double mod_value)
{
    if (std::abs(mod_value) < std::numeric_limits<double>::epsilon() * EPSILON_SCALE)
    {
        throw std::domain_error("[Math] Modulus Function failed: modulus value cannot be 0 or (~0).");
    }

    return x.Apply([=](double value) {
        return std::fmod(value, mod_value);
        });
}

template <MathContainer T>
T Math::Power(const T& x, double exponent)
{
    return x.Apply([=](double value) {
        if (value < 0.0 && std::fmod(exponent, 1.0) != 0.0)
        {
            throw std::domain_error("[Math] Power Function failed: negative base detected with non-integer exponent -results non-real number.");
        }

        return std::pow(value, exponent);
        });
}

template <MathContainer T>
T Math::Round(const T& x, int decimal_place)
{
    if (decimal_place < 0)
        throw std::invalid_argument("[Math] Round Function failed: decimal_place cannot be negative.");

    const double power_of_10 = std::pow(10.0, decimal_place);

    return x.Apply([=](double value) {
        return std::round(value * power_of_10) / power_of_10;
        });
}

template <MathContainer T>
T Math::Sqrt(const T& x)
{
    return x.Apply([](double value) {
        if (value < 0.0)
        {
            throw std::domain_error("[Math] Sqrt Function failed: negative value found in input, " + std::to_string(value));
        }

        return std::sqrt(value);
        });
}

// ========================================
// Trigonometric Methods
// ========================================
template <MathContainer T>
T Math::Sin(const T& x)
{
    return x.Apply([](double value) {
        return std::sin(value);
        });
}

template <MathContainer T>
T Math::Cos(const T& x)
{
    return x.Apply([](double value) {
        return std::cos(value);
        });
}

template <MathContainer T>
T Math::Tan(const T& x)
{
    return x.Apply([](double value) {
        double cos_value = std::cos(value);

        if (std::abs(cos_value) < std::numeric_limits<double>::epsilon() * EPSILON_SCALE)
        {
            throw std::domain_error("[Math] Tangent Function failed: undefined near odd multiples of pi/2.");
        }

        return std::tan(value);
        });
}

template <MathContainer T>
T Math::Csc(const T& x)
{
    return x.Apply([](double value) {
        double sin_value = std::sin(value);

        if (std::abs(sin_value) < std::numeric_limits<double>::epsilon() * EPSILON_SCALE)
        {
            throw std::domain_error("[Math] Cosecant Function failed: undefined near multiples of pi.");
        }

        return 1.0 / sin_value;
        });
}

template <MathContainer T>
T Math::Sec(const T& x)
{
    return x.Apply([](double value) {
        double cos_value = std::cos(value);

        if (std::abs(cos_value) < std::numeric_limits<double>::epsilon() * EPSILON_SCALE)
        {
            throw std::domain_error("[Math] Secant Function failed: undefined near odd multiples of pi/2.");
        }

        return 1.0 / cos_value;
        });
}

template <MathContainer T>
T Math::Cot(const T& x)
{
    return x.Apply([](double value) {
        double sin_value = std::sin(value);

        if (std::abs(sin_value) < std::numeric_limits<double>::epsilon() * EPSILON_SCALE)
        {
            throw std::domain_error("[Math] Cotangent Function failed: undefined near multiples of pi.");
        }

        return std::cos(value) / sin_value;
        });
}

// ========================================
// Inverse Trigonometric Methods
// ========================================
template <MathContainer T>
T Math::Asin(const T& x)
{
    return x.Apply([](double value) {
        if (value < -1.0 || value > 1.0)
        {
            throw std::domain_error("[Math] Arc-sine Function failed: arcsine is only defined for values in [-1, 1].");
        }

        return std::asin(value);
        });
}

template <MathContainer T>
T Math::Acos(const T& x)
{
    return x.Apply([](double value) {
        if (value < -1.0 || value > 1.0)
        {
            throw std::domain_error("[Math] Arc-cosine Function failed: arccosine is only defined for values in [-1, 1].");
        }

        return std::acos(value);
        });
}

template <MathContainer T>
T Math::Atan(const T& x)
{
    return x.Apply([](double value) {
        return std::atan(value);
        });
}

template <MathContainer T>
T Math::Acsc(const T& x)
{
    return x.Apply([](double value) {
        if (value > -1.0 && value < 1.0)
        {
            throw std::domain_error("[Math] Arc-cosecant Function failed: arccosecant is not defined for values in (-1, 1).");
        }

        return std::asin(1.0 / value);
        });
}

template <MathContainer T>
T Math::Asec(const T& x)
{
    return x.Apply([](double value) {
        if (value > -1.0 && value < 1.0)
        {
            throw std::domain_error("[Math] Arc-secant Function failed: arcsecant is not defined for values in (-1, 1).");
        }

        return std::acos(1.0 / value);
        });
}

template <MathContainer T>
T Math::Acot(const T& x)
{
    return x.Apply([](double value) {
        if (std::abs(value) < std::numeric_limits<double>::epsilon() * Math::EPSILON_SCALE)
        {
            return std::numbers::pi / 2.0;
        }
        else
        {
            double acot_value = std::atan(1.0 / value);
            acot_value += (value < 0.0) ? std::numbers::pi : 0.0;
            return acot_value;
        }
        });
}

// ========================================
// Hyperbolic Methods
// ========================================
template <MathContainer T>
T Math::Sinh(const T& x)
{
    return x.Apply([](double value) {
        return std::sinh(value);
        });
}

template <MathContainer T>
T Math::Cosh(const T& x)
{
    return x.Apply([](double value) {
        return std::cosh(value);
        });
}

template <MathContainer T>
T Math::Tanh(const T& x)
{
    return x.Apply([](double value) {
        return std::tanh(value);
        });
}

template <MathContainer T>
T Math::Csch(const T& x)
{
    return x.Apply([](double value) {
        double sinh_value = std::sinh(value);

        if (std::abs(sinh_value) < std::numeric_limits<double>::epsilon() * Math::EPSILON_SCALE)
        {
            throw std::domain_error("[Math] Hyperbolic Cosecant Function failed: hyperbolic cosecant is undefined at ~ zero.");
        }

        return 1.0 / sinh_value;
        });
}

template <MathContainer T>
T Math::Sech(const T& x)
{
    return x.Apply([](double value) {
        return 1.0 / std::cosh(value);
        });
}

template <MathContainer T>
T Math::Coth(const T& x)
{
    return x.Apply([](double value) {
        double sinh_value = std::sinh(value);

        if (std::abs(sinh_value) < std::numeric_limits<double>::epsilon() * Math::EPSILON_SCALE)
        {
            throw std::domain_error("[Math] Hyperbolic Cotangent Function failed: hyperbolic cotangent is undefined at ~ zero.");
        }

        return std::cosh(value) / sinh_value;
        });
}

// ========================================
// Inverse Hyperbolic Methods
// ========================================
template <MathContainer T>
T Math::Asinh(const T& x)
{
    return x.Apply([](double value) {
        return std::asinh(value);
        });
}

template <MathContainer T>
T Math::Acosh(const T& x)
{
    return x.Apply([](double value) {
        if (value < 1.0)
        {
            throw std::domain_error("[Math] Inverse Hyperbolic Cosine Function failed: inverse hyperbolic cosine is only defined for values >= 1.");
        }

        return std::acosh(value);
        });
}

template <MathContainer T>
T Math::Atanh(const T& x)
{
    return x.Apply([](double value) {
        if (value <= -1.0 || value >= 1.0)
        {
            throw std::domain_error("[Math] Inverse Hyperbolic Tangent Function failed: inverse hyperbolic tangent is only defined for values in (-1, 1).");
        }

        return std::atanh(value);
        });
}

template <MathContainer T>
T Math::Acsch(const T& x)
{
    return x.Apply([](double value) {
        if (std::abs(value) < std::numeric_limits<double>::epsilon() * Math::EPSILON_SCALE)
        {
            throw std::domain_error("[Math] Inverse Hyperbolic Cosecant Function failed: inverse hyperbolic cosecant is undefined at zero.");
        }

        return std::asinh(1.0 / value);
        });
}

template <MathContainer T>
T Math::Asech(const T& x)
{
    return x.Apply([](double value) {
        if (value <= 0.0 || value > 1.0)
        {
            throw std::domain_error("[Math] Inverse Hyperbolic Secant Function failed: inverse hyperbolic secant is only defined for values in (0, 1].");
        }

        return std::acosh(1.0 / value);
        });
}

template <MathContainer T>
T Math::Acoth(const T& x)
{
    return x.Apply([](double value) {
        if (value >= -1.0 && value <= 1.0)
        {
            throw std::domain_error("[Math] Inverse Hyperbolic Cotangent Function failed: inverse hyperbolic cotangent is not defined for values in [-1, 1].");
        }

        return std::atanh(1.0 / value);
        });
}
