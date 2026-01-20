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
std::vector<T> Utils::ScaleNShift(const std::vector<T>& nums, const std::vector<U>& scale, const std::vector<V>& shift)
{
    if (nums.size() != scale.size())
    {
        throw std::invalid_argument("[Tensor-Utils] Scaling-Shifting failed: size mismatch between nums array and scale array.");
    }

    if (nums.size() != shift.size())
    {
        throw std::invalid_argument("[Tensor-Utils] Scaling-Shifting failed: size mismatch between nums array and shift array.");
    }

    size_t n = nums.size();
    std::vector<T> result(n);

    for (size_t i = 0; i < n; i++)
    {
        result[i] = static_cast<T>(nums[i] * static_cast<T>(scale[i]) + static_cast<T>(shift[i]));
    }

    return result;
}

template<Numeric T, Numeric U, Numeric V>
std::vector<T> Utils::ScaleNShift(const std::vector<T>& nums, const std::vector<U>& scale, V shift)
{
    if (nums.size() != scale.size())
    {
        throw std::invalid_argument("[Tensor-Utils] Scaling-Shifting failed: size mismatch between nums array and scale array.");
    }

    size_t n = nums.size();
    std::vector<T> result(n);

    for (size_t i = 0; i < n; i++)
    {
        result[i] = static_cast<T>(nums[i] * static_cast<T>(scale[i]) + static_cast<T>(shift));
    }

    return result;
}

template<Numeric T, Numeric U, Numeric V>
std::vector<T> Utils::ScaleNShift(const std::vector<T>& nums, U scale, const std::vector<V>& shift)
{
    if (nums.size() != shift.size())
    {
        throw std::invalid_argument("[Tensor-Utils] Scaling-Shifting failed: size mismatch between nums array and shift array.");
    }

    size_t n = nums.size();
    std::vector<T> result(n);

    for (size_t i = 0; i < n; i++)
    {
        result[i] = static_cast<T>(nums[i] * static_cast<T>(scale) + static_cast<T>(shift[i]));
    }

    return result;
}

template<Numeric T, Numeric U, Numeric V>
std::vector<T> Utils::ScaleNShift(const std::vector<T>& nums, U scale, V shift)
{
    size_t n = nums.size();
    std::vector<T> result(n);

    for (size_t i = 0; i < n; i++)
    {
        result[i] = static_cast<T>(nums[i] * static_cast<T>(scale) + static_cast<T>(shift));
    }

    return result;
}


// ========================================
// Vector L1 Norm Function
// ========================================
template<Numeric T>
double Utils::Norm(const std::vector<T>& nums)
{
    double norm = 0.0;
    for (const auto& value : nums)
    {
        norm += (value * value);
    }

    norm = std::sqrt(norm);
    return norm;
}


// ========================================
// Vector & Matrix Operation Function(s)
// ========================================
template<Numeric T>
bool Utils::IsRectangular(const std::vector<std::vector<T>>& matrix)
{
    static_assert(std::is_arithmetic_v<T>, "IsRectangular requires a numeric type");

    if (matrix.empty())
    {
        return true;
    }

    size_t columns = matrix[0].size();

    for (const auto& row : matrix)
    {
        if (row.size() != columns)
        {
            return false;
        }
    }
    return true;
}

template<Numeric T>
std::vector<T> Utils::MatrixToVector(const std::vector<std::vector<T>>& matrix)
{
    static_assert(std::is_arithmetic_v<T>, "MatrixToVector requires a numeric type");

    if (matrix.empty())
    {
        return {};
    }

    size_t rows = matrix.size();
    size_t columns = matrix[0].size();

    std::vector<T> nums;
    nums.reserve(rows * columns);

    for (size_t i = 0; i < rows; i++)
    {
        nums.insert(nums.end(), matrix[i].begin(), matrix[i].end());
    }

    return nums;
}

template<Numeric T>
std::vector<std::vector<T>> Utils::VectorToMatrix(const std::vector<T>& vec, std::pair<int, int> shape)
{
    static_assert(std::is_arithmetic_v<T>, "VectorToMatrix requires a numeric type");

    long long expected = static_cast<long long>(shape.first) * static_cast<long long>(shape.second);
    if (expected != static_cast<long long>(vec.size()))
    {
        throw std::invalid_argument("[Tensor-Utils] Vector Conversion to Matrix failed: vector size mismatch with volume of shape.");
    }

    std::vector<std::vector<T>> matrix(shape.first, std::vector<T>(shape.second));

    for (size_t i = 0; i < vec.size(); i++)
    {
        size_t k1 = i / static_cast<size_t>(shape.second);
        size_t k2 = i % static_cast<size_t>(shape.second);
        matrix[k1][k2] = vec[i];
    }

    return matrix;
}
