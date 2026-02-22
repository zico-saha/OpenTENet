#include "Utils.h"

// ========================================
// Validation Function(s)
// ========================================
template<Numeric T>
bool Utils::IsAllPositive(const std::vector<T>& nums)
{
    static_assert(std::is_arithmetic_v<T>, "IsAllPositive requires a numeric type");

    for (const auto& num : nums)
    {
        if (!(num > static_cast<T>(0)))
        {
            return false;
        }
    }
    return true;
}

template<Numeric T>
bool Utils::IsAnyNegative(const std::vector<T>& nums)
{
    static_assert(std::is_arithmetic_v<T>, "IsAnyNegative requires a numeric type");

    for (const auto& num : nums)
    {
        if (num < static_cast<T>(0))
        {
            return true;
        }
    }
    return false;
}

template<Numeric T>
bool Utils::IsAllUnique(const std::vector<T>& nums)
{
    static_assert(std::is_arithmetic_v<T>, "IsAllUnique requires a numeric type");

    std::set<T> freq;
    for (const auto& num : nums)
    {
        if (freq.count(num))
        {
            return false;
        }
        freq.insert(num);
    }
    return true;
}

template<Numeric T>
bool Utils::IsValidData(const std::vector<T>& nums)
{
    static_assert(std::is_arithmetic_v<T>, "IsValidData requires a numeric type");

    if constexpr (std::is_floating_point_v<T>)
    {
        for (const auto& num : nums)
        {
            if (std::isnan(num) || std::isinf(num))
            {
                return false;
            }
        }
    }
    else
    {
        (void)nums;
    }
    return true;
}


// ========================================
// Bounds Checking Function(s)
// ========================================
template<Numeric T, Numeric U, Numeric V>
bool Utils::IsBounded(const std::vector<T>& nums, U upper_bound, V lower_bound, bool strict)
{
    for (const auto& num : nums)
    {
        if (num < static_cast<T>(lower_bound) || num > static_cast<T>(upper_bound))
        {
            return false;
        }
        if (strict && (num <= static_cast<T>(lower_bound) || num >= static_cast<T>(upper_bound)))
        {
            return false;
        }
    }
    return true;
}

template<Numeric T, Numeric U, Numeric V>
bool Utils::IsBounded(const std::vector<T>& nums, const std::vector<U>& upper_bounds, const std::vector<V>& lower_bounds, bool strict)
{
    if (nums.size() != upper_bounds.size())
    {
        throw std::invalid_argument("[Tensor-Utils] Bounds Checking failed: array size mismatch with `upper_bounds`.");
    }

    if (nums.size() != lower_bounds.size())
    {
        throw std::invalid_argument("[Tensor-Utils] Bounds Checking failed: array size mismatch with `lower_bounds`.");
    }

    for (size_t i = 0; i < nums.size(); i++)
    {
        if (nums[i] < static_cast<T>(lower_bounds[i]) || nums[i] > static_cast<T>(upper_bounds[i]))
        {
            return false;
        }
        if (strict && (nums[i] <= static_cast<T>(lower_bounds[i]) || nums[i] >= static_cast<T>(upper_bounds[i])))
        {
            return false;
        }
    }
    return true;
}

template<Numeric T, Numeric U, Numeric V>
bool Utils::IsBounded(const std::vector<T>& nums, const std::vector<U>& upper_bounds, V lower_bound, bool strict)
{
    if (nums.size() != upper_bounds.size())
    {
        throw std::invalid_argument("[Tensor-Utils] Bounds Checking failed: array size mismatch with `upper_bounds`.");
    }

    for (size_t i = 0; i < nums.size(); i++)
    {
        if (nums[i] < static_cast<T>(lower_bound) || nums[i] > static_cast<T>(upper_bounds[i]))
        {
            return false;
        }
        if (strict && (nums[i] <= static_cast<T>(lower_bound) || nums[i] >= static_cast<T>(upper_bounds[i])))
        {
            return false;
        }
    }
    return true;
}

template<Numeric T, Numeric U, Numeric V>
bool Utils::IsBounded(const std::vector<T>& nums, U upper_bound, const std::vector<V>& lower_bounds, bool strict)
{
    if (nums.size() != lower_bounds.size())
    {
        throw std::invalid_argument("[Tensor-Utils] Bounds Checking failed: array size mismatch with `lower_bounds`.");
    }

    for (size_t i = 0; i < nums.size(); i++)
    {
        if (nums[i] < static_cast<T>(lower_bounds[i]) || nums[i] > static_cast<T>(upper_bound))
        {
            return false;
        }
        if (strict && (nums[i] <= static_cast<T>(lower_bounds[i]) || nums[i] >= static_cast<T>(upper_bound)))
        {
            return false;
        }
    }
    return true;
}

template<Numeric T, Numeric U>
bool Utils::IsUpperBounded(const std::vector<T>& nums, U upper_bound, bool strict)
{
    for (const auto& num : nums)
    {
        if (num > static_cast<T>(upper_bound))
        {
            return false;
        }
        if (strict && num >= static_cast<T>(upper_bound))
        {
            return false;
        }
    }
    return true;
}

template<Numeric T, Numeric U>
bool Utils::IsUpperBounded(const std::vector<T>& nums, const std::vector<U>& upper_bounds, bool strict)
{
    if (nums.size() != upper_bounds.size())
    {
        throw std::invalid_argument("[Tensor-Utils] Bounds Checking failed: array size mismatch with `upper_bounds`.");
    }

    for (size_t i = 0; i < nums.size(); i++)
    {
        if (nums[i] > static_cast<T>(upper_bounds[i]))
        {
            return false;
        }
        if (strict && nums[i] >= static_cast<T>(upper_bounds[i]))
        {
            return false;
        }
    }
    return true;
}

template<Numeric T, Numeric U>
bool Utils::IsLowerBounded(const std::vector<T>& nums, U lower_bound, bool strict)
{
    for (const auto& num : nums)
    {
        if (num < static_cast<T>(lower_bound))
        {
            return false;
        }
        if (strict && num <= static_cast<T>(lower_bound))
        {
            return false;
        }
    }
    return true;
}

template<Numeric T, Numeric U>
bool Utils::IsLowerBounded(const std::vector<T>& nums, const std::vector<U>& lower_bounds, bool strict)
{
    if (nums.size() != lower_bounds.size())
    {
        throw std::invalid_argument("[Tensor-Utils] Bounds Checking failed: array size mismatch with `lower_bounds`.");
    }

    for (size_t i = 0; i < nums.size(); i++)
    {
        if (nums[i] < static_cast<T>(lower_bounds[i]))
        {
            return false;
        }
        if (strict && nums[i] <= static_cast<T>(lower_bounds[i]))
        {
            return false;
        }
    }
    return true;
}


// ========================================
// Permutation Function(s)
// ========================================
template<Numeric T>
std::vector<T> Utils::Permute(const std::vector<T>& nums, const std::vector<int>& permutation)
{
    static_assert(std::is_arithmetic_v<T>, "Permute requires a numeric type");

    if (nums.size() != permutation.size())
    {
        throw std::invalid_argument("[Tensor-Utils] Permutation failed: array size mismatch with permutation.");
    }

    if (!Utils::IsBounded(permutation, static_cast<int>(nums.size()), -1, true))
    {
        throw std::invalid_argument("[Tensor-Utils] Permutation failed: permutation values are not bounded.");
    }

    if (!Utils::IsAllUnique(permutation))
    {
        throw std::invalid_argument("[Tensor-Utils] Permutation failed: duplicate values found in permutation array.");
    }

    size_t n = nums.size();
    std::vector<T> result(n);

    for (size_t i = 0; i < n; i++)
    {
        result[i] = nums[static_cast<size_t>(permutation[i])];
    }

    return result;
}


// ========================================
// Signal Transformation Function(s)
// ========================================
template<Numeric T, Numeric U, Numeric V>
std::vector<T> Utils::ScaleNShift(const std::vector<T>& _nums, const std::vector<U>& _scale, const std::vector<V>& _shift)
{
    if (_nums.size() != _scale.size())
    {
        throw std::invalid_argument("[Utils] Scaling-Shifting failed: size mismatch between nums array and scale array.");
    }

    if (_nums.size() != _shift.size())
    {
        throw std::invalid_argument("[Utils] Scaling-Shifting failed: size mismatch between nums array and shift array.");
    }

    size_t n = _nums.size();
    std::vector<T> result(n);

    for (size_t i = 0; i < n; i++)
    {
        result[i] = static_cast<T>(_nums[i] * static_cast<T>(_scale[i]) + static_cast<T>(_shift[i]));
    }

    return result;
}

template<Numeric T, Numeric U, Numeric V>
std::vector<T> Utils::ScaleNShift(const std::vector<T>& _nums, const std::vector<U>& _scale, V _shift)
{
    if (_nums.size() != _scale.size())
    {
        throw std::invalid_argument("[Utils] Scaling-Shifting failed: size mismatch between nums array and scale array.");
    }

    size_t n = _nums.size();
    std::vector<T> result(n);

    for (size_t i = 0; i < n; i++)
    {
        result[i] = static_cast<T>(_nums[i] * static_cast<T>(_scale[i]) + static_cast<T>(_shift));
    }

    return result;
}

template<Numeric T, Numeric U, Numeric V>
std::vector<T> Utils::ScaleNShift(const std::vector<T>& _nums, U _scale, const std::vector<V>& _shift)
{
    if (_nums.size() != _shift.size())
    {
        throw std::invalid_argument("[Utils] Scaling-Shifting failed: size mismatch between nums array and shift array.");
    }

    size_t n = _nums.size();
    std::vector<T> result(n);

    for (size_t i = 0; i < n; i++)
    {
        result[i] = static_cast<T>(_nums[i] * static_cast<T>(_scale) + static_cast<T>(_shift[i]));
    }

    return result;
}

template<Numeric T, Numeric U, Numeric V>
std::vector<T> Utils::ScaleNShift(const std::vector<T>& _nums, U _scale, V _shift)
{
    size_t n = _nums.size();
    std::vector<T> result(n);

    for (size_t i = 0; i < n; i++)
    {
        result[i] = static_cast<T>(_nums[i] * static_cast<T>(_scale) + static_cast<T>(_shift));
    }

    return result;
}

// ========================================
// Vector Norm Function(s)
// ========================================
template<Numeric T, Numeric P>
double Utils::Norm(const std::vector<T>& _nums, const P& _p)
{
    if (_nums.empty())
    {
        return 0.0;
    }

    if (static_cast<double>(_p) < 1.0)
    {
        throw std::invalid_argument("[Utils] Norm Computation failed: p-type should be >= 1 for valid and non-quasi norm.");
    }

    double maxVal = 0.0;
    for (const auto& value : _nums)
    {
        maxVal = std::max(maxVal, std::abs(static_cast<double>(value)));
    }

    if (std::abs(maxVal) <= 1e-9)
    {
        return 0.0;
    }

    double sum = 0.0;
    for (const auto& value : _nums)
    {
        double scaled = std::abs(static_cast<double>(value)) / maxVal;
        sum += std::pow(scaled, _p);
    }

    return maxVal * std::pow(sum, 1.0 / _p);
}

template<Numeric T>
double Utils::InfinityNorm(const std::vector<T>& _nums)
{
    if (_nums.empty())
    {
        return 0.0;
    }

    double maxVal = 0.0;

    for (const auto& value : _nums)
    {
        maxVal = std::max(maxVal, std::abs(static_cast<double>(value)));
    }

    return maxVal;
}

// ========================================
// Vector & Matrix Operation Function(s)
// ========================================
template<Numeric T>
bool Utils::IsRectangular(const std::vector<std::vector<T>>& _matrix)
{
    static_assert(std::is_arithmetic_v<T>, "IsRectangular requires a numeric type");

    if (_matrix.empty())
    {
        return true;
    }

    size_t columns = _matrix[0].size();

    for (const auto& row : _matrix)
    {
        if (row.size() != columns)
        {
            return false;
        }
    }
    return true;
}
