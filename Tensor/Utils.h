#pragma once

#include <vector>
#include <set>
#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <cmath>
#include <climits>
#include <cassert>
#include <utility>
#include <functional>
#include <stdexcept>
#include <type_traits>

template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

namespace Utils
{
    /**
     * @brief Checks if all elements in a vector are positive.
     *
     * @tparam T Numeric type satisfying the Numeric concept
     * @param nums Input vector to validate
     * 
     * @return true if all elements are greater than zero
     * @return false if any element is zero or negative
     *
     * @note Empty vectors return true (vacuous truth)
     * @note Uses strict comparison (num > 0), so zero is not considered positive
     */
    template<Numeric T>
    bool IsAllPositive(const std::vector<T>& nums);

    /**
     * @brief Checks if any element in a vector is negative.
     *
     * @tparam T Numeric type satisfying the Numeric concept
     * @param nums Input vector to validate
     * 
     * @return true if at least one element is less than zero
     * @return false if all elements are non-negative
     *
     * @note Empty vectors return false
     * @note Zero is not considered negative
     */
    template<Numeric T>
    bool IsAnyNegative(const std::vector<T>& nums);

    /**
     * @brief Checks if all elements in a vector are unique (no duplicates).
     *
     * @tparam T Numeric type satisfying the Numeric concept
     * @param nums Input vector to validate
     * 
     * @return true if all elements are distinct
     * @return false if any duplicates exist
     *
     * @note Empty vectors and single-element vectors return true
     * @note Uses std::set for duplicate detection (O(n log n) complexity)
     */
    template<Numeric T>
    bool IsAllUnique(const std::vector<T>& nums);

    /**
     * @brief Validates floating-point data for NaN and infinity values.
     *
     * @tparam T Numeric type satisfying the Numeric concept
     * @param nums Input vector to validate
     * 
     * @return true if all floating-point values are finite (not NaN or infinity)
     * @return false if any NaN or infinity is found
     *
     * @note For integer types, always returns true (no-op)
     * @note Empty vectors return true
     */
    template<Numeric T>
    bool IsValidData(const std::vector<T>& nums);
    
    /**
     * @brief Checks if all vector elements are within specified scalar bounds.
     *
     * @tparam T Numeric type of the input vector
     * @tparam U Numeric type of the upper bound
     * @tparam V Numeric type of the lower bound
     * @param nums Input vector to validate
     * @param upper_bound Maximum allowed value (inclusive by default)
     * @param lower_bound Minimum allowed value (inclusive by default)
     * @param strict If true, uses exclusive bounds (< and >); if false, uses inclusive bounds (<= and >=)
     * 
     * @return true if all elements satisfy the bound conditions
     * @return false if any element violates the bounds
     *
     * @note Empty vectors return true
     * @note Types are automatically converted to T for comparison
     */
    template<Numeric T, Numeric U, Numeric V>
    bool IsBounded(const std::vector<T>& nums, U upper_bound, V lower_bound, bool strict = false);

    /**
     * @brief Checks if vector elements are within element-wise vector bounds.
     *
     * @tparam T Numeric type of the input vector
     * @tparam U Numeric type of the upper bounds vector
     * @tparam V Numeric type of the lower bounds vector
     * @param nums Input vector to validate
     * @param upper_bounds Vector of maximum allowed values (one per element)
     * @param lower_bounds Vector of minimum allowed values (one per element)
     * @param strict If true, uses exclusive bounds; if false, uses inclusive bounds
     * 
     * @return true if all elements satisfy their respective bound conditions
     * @return false if any element violates its bounds
     *
     * @throws std::invalid_argument if size mismatch between nums and upper_bounds
     * @throws std::invalid_argument if size mismatch between nums and lower_bounds
     *
     * @note All three vectors must have the same size
     */
    template<Numeric T, Numeric U, Numeric V>
    bool IsBounded(const std::vector<T>& nums, const std::vector<U>& upper_bounds, const std::vector<V>& lower_bounds, bool strict = false);

    /**
     * @brief Checks if vector elements are within element-wise upper bounds and scalar lower bound.
     *
     * @tparam T Numeric type of the input vector
     * @tparam U Numeric type of the upper bounds vector
     * @tparam V Numeric type of the lower bound
     * @param nums Input vector to validate
     * @param upper_bounds Vector of maximum allowed values (one per element)
     * @param lower_bound Scalar minimum allowed value (applied to all elements)
     * @param strict If true, uses exclusive bounds; if false, uses inclusive bounds
     * 
     * @return true if all elements satisfy the bound conditions
     * @return false if any element violates the bounds
     *
     * @throws std::invalid_argument if size mismatch between nums and upper_bounds
     */
    template<Numeric T, Numeric U, Numeric V>
    bool IsBounded(const std::vector<T>& nums, const std::vector<U>& upper_bounds, V lower_bound, bool strict = false);

    /**
     * @brief Checks if vector elements are within scalar upper bound and element-wise lower bounds.
     *
     * @tparam T Numeric type of the input vector
     * @tparam U Numeric type of the upper bound
     * @tparam V Numeric type of the lower bounds vector
     * @param nums Input vector to validate
     * @param upper_bound Scalar maximum allowed value (applied to all elements)
     * @param lower_bounds Vector of minimum allowed values (one per element)
     * @param strict If true, uses exclusive bounds; if false, uses inclusive bounds
     * 
     * @return true if all elements satisfy the bound conditions
     * @return false if any element violates the bounds
     *
     * @throws std::invalid_argument if size mismatch between nums and lower_bounds
     */
    template<Numeric T, Numeric U, Numeric V>
    bool IsBounded(const std::vector<T>& nums, U upper_bound, const std::vector<V>& lower_bounds, bool strict = false);

    /**
     * @brief Checks if all vector elements are below or equal to a scalar upper bound.
     *
     * @tparam T Numeric type of the input vector
     * @tparam U Numeric type of the upper bound
     * @param nums Input vector to validate
     * @param upper_bound Maximum allowed value
     * @param strict If true, uses exclusive bound (<); if false, uses inclusive bound (<=)
     * 
     * @return true if all elements satisfy the upper bound condition
     * @return false if any element exceeds the upper bound
     *
     * @note Empty vectors return true
     */
    template<Numeric T, Numeric U>
    bool IsUpperBounded(const std::vector<T>& nums, U upper_bound, bool strict = false);

    /**
     * @brief Checks if vector elements are below or equal to element-wise upper bounds.
     *
     * @tparam T Numeric type of the input vector
     * @tparam U Numeric type of the upper bounds vector
     * @param nums Input vector to validate
     * @param upper_bounds Vector of maximum allowed values (one per element)
     * @param strict If true, uses exclusive bounds (<); if false, uses inclusive bounds (<=)
     * 
     * @return true if all elements satisfy their respective upper bound
     * @return false if any element exceeds its upper bound
     *
     * @throws std::invalid_argument if size mismatch between nums and upper_bounds
     */
    template<Numeric T, Numeric U>
    bool IsUpperBounded(const std::vector<T>& nums, const std::vector<U>& upper_bounds, bool strict = false);

    /**
     * @brief Checks if all vector elements are above or equal to a scalar lower bound.
     *
     * @tparam T Numeric type of the input vector
     * @tparam U Numeric type of the lower bound
     * @param nums Input vector to validate
     * @param lower_bound Minimum allowed value
     * @param strict If true, uses exclusive bound (>); if false, uses inclusive bound (>=)
     * 
     * @return true if all elements satisfy the lower bound condition
     * @return false if any element is below the lower bound
     *
     * @note Empty vectors return true
     */
    template<Numeric T, Numeric U>
    bool IsLowerBounded(const std::vector<T>& nums, U lower_bound, bool strict = false);

    /**
     * @brief Checks if vector elements are above or equal to element-wise lower bounds.
     *
     * @tparam T Numeric type of the input vector
     * @tparam U Numeric type of the lower bounds vector
     * @param nums Input vector to validate
     * @param lower_bounds Vector of minimum allowed values (one per element)
     * @param strict If true, uses exclusive bounds (>); if false, uses inclusive bounds (>=)
     * 
     * @return true if all elements satisfy their respective lower bound
     * @return false if any element is below its lower bound
     *
     * @throws std::invalid_argument if size mismatch between nums and lower_bounds
     */
    template<Numeric T, Numeric U>
    bool IsLowerBounded(const std::vector<T>& nums, const std::vector<U>& lower_bounds, bool strict = false);
    
    /**
     * @brief Reorders vector elements according to a permutation mapping.
     *
     * @tparam T Numeric type satisfying the Numeric concept
     * @param nums Input vector to permute
     * @param permutation Index mapping vector (must contain unique values in range [0, nums.size()))
     * 
     * @return std::vector<T> Permuted vector where result[i] = nums[permutation[i]]
     *
     * @throws std::invalid_argument if size mismatch between nums and permutation
     * @throws std::invalid_argument if permutation contains out-of-bounds indices
     * @throws std::invalid_argument if permutation contains duplicate indices
     *
     * @note permutation must be a valid permutation (bijective mapping)
     * @note Example: nums = {10, 20, 30}, permutation = {2, 0, 1} ? result = {30, 10, 20}
     */
    template<Numeric T>
    std::vector<T> Permute(const std::vector<T>& nums, const std::vector<int>& permutation);

    /**
     * @brief Applies element-wise scaling and shifting: result[i] = nums[i] * scale[i] + shift[i].
     *
     * @tparam T Numeric type of the input vector
     * @tparam U Numeric type of the scale vector
     * @tparam V Numeric type of the shift vector
     * @param nums Input vector
     * @param scale Scaling factors (one per element)
     * @param shift Shift values (one per element)
     * 
     * @return std::vector<T> Transformed vector with type T
     *
     * @throws std::invalid_argument if size mismatch between nums and scale
     * @throws std::invalid_argument if size mismatch between nums and shift
     *
     * @note All three vectors must have the same size
     * @note Result type matches the input vector type T
     */
    template<Numeric T, Numeric U, Numeric V>
    std::vector<T> ScaleNShift(const std::vector<T>& nums, const std::vector<U>& scale, const std::vector<V>& shift);

    /**
     * @brief Applies element-wise scaling with scalar shift: result[i] = nums[i] * scale[i] + shift.
     *
     * @tparam T Numeric type of the input vector
     * @tparam U Numeric type of the scale vector
     * @tparam V Numeric type of the shift scalar
     * @param nums Input vector
     * @param scale Scaling factors (one per element)
     * @param shift Scalar shift value (applied to all elements)
     * 
     * @return std::vector<T> Transformed vector with type T
     *
     * @throws std::invalid_argument if size mismatch between nums and scale
     */
    template<Numeric T, Numeric U, Numeric V>
    std::vector<T> ScaleNShift(const std::vector<T>& nums, const std::vector<U>& scale, V shift);

    /**
     * @brief Applies scalar scaling with element-wise shift: result[i] = nums[i] * scale + shift[i].
     *
     * @tparam T Numeric type of the input vector
     * @tparam U Numeric type of the scale scalar
     * @tparam V Numeric type of the shift vector
     * @param nums Input vector
     * @param scale Scalar scaling factor (applied to all elements)
     * @param shift Shift values (one per element)
     * 
     * @return std::vector<T> Transformed vector with type T
     *
     * @throws std::invalid_argument if size mismatch between nums and shift
     */
    template<Numeric T, Numeric U, Numeric V>
    std::vector<T> ScaleNShift(const std::vector<T>& nums, U scale, const std::vector<V>& shift);

    /**
     * @brief Applies scalar scaling and shifting: result[i] = nums[i] * scale + shift.
     *
     * @tparam T Numeric type of the input vector
     * @tparam U Numeric type of the scale scalar
     * @tparam V Numeric type of the shift scalar
     * @param nums Input vector
     * @param scale Scalar scaling factor (applied to all elements)
     * @param shift Scalar shift value (applied to all elements)
     * 
     * @return std::vector<T> Transformed vector with type T
     */
    template<Numeric T, Numeric U, Numeric V>
    std::vector<T> ScaleNShift(const std::vector<T>& nums, U scale, V shift);

    /**
     * @brief Checks if a 2D matrix has consistent row lengths (is rectangular).
     *
     * @tparam T Numeric type satisfying the Numeric concept
     * @param matrix Input 2D vector to validate
     * 
     * @return true if all rows have the same length
     * @return false if row lengths differ
     *
     * @note Empty matrices return true
     * @note A matrix with one row is considered rectangular
     */
    template<Numeric T>
    bool IsRectangular(const std::vector<std::vector<T>>& matrix);

    /**
     * @brief Transposes a matrix (swaps rows and columns).
     *
     * @tparam T Numeric type satisfying the Numeric concept
     * @param matrix Input matrix to transpose (MxN)
     * 
     * @return std::vector<std::vector<T>> Transposed matrix (NxM)
     *
     * @throws std::invalid_argument if input matrix is not rectangular
     *
     * @note Returns empty matrix if input is empty
     * @note Result[i][j] = matrix[j][i]
     */
    template<Numeric T>
    std::vector<std::vector<T>> TransposeMatrix(const std::vector<std::vector<T>>& matrix);

    /**
     * @brief Performs standard matrix multiplication.
     *
     * @tparam T Numeric type satisfying the Numeric concept
     * @param m1 First matrix (MxK)
     * @param m2 Second matrix (KxN)
     * 
     * @return std::vector<std::vector<T>> Product matrix (MxN)
     *
     * @throws std::invalid_argument if either input matrix is empty
     * @throws std::invalid_argument if matrix dimensions are incompatible (m1 columns != m2 rows)
     * @throws std::invalid_argument if first matrix is not rectangular
     * @throws std::invalid_argument if second matrix is not rectangular
     *
     * @note Complexity: O(M * N * K)
     * @note Result[i][j] = sum(m1[i][k] * m2[k][j]) for all k
     */
    template<Numeric T>
    std::vector<std::vector<T>> StandardMatrixMultiply(const std::vector<std::vector<T>>& m1, const std::vector<std::vector<T>>& m2);

    /**
     * @brief Flattens a 2D matrix into a 1D vector (row-major order).
     *
     * @tparam T Numeric type satisfying the Numeric concept
     * @param matrix Input matrix to flatten
     * 
     * @return std::vector<T> Flattened vector containing all elements in row-major order
     *
     * @note Returns empty vector if input is empty
     * @note Elements are concatenated: [row0 elements, row1 elements, ...]
     */
    template<Numeric T>
    std::vector<T> MatrixToVector(const std::vector<std::vector<T>>& matrix);

    /**
     * @brief Reshapes a 1D vector into a 2D matrix with specified dimensions.
     *
     * @tparam T Numeric type satisfying the Numeric concept
     * @param vec Input vector to reshape
     * @param shape Target matrix dimensions as (rows, columns)
     * 
     * @return std::vector<std::vector<T>> Reshaped matrix
     *
     * @throws std::invalid_argument if vector size doesn't match rows * columns
     *
     * @note Elements are filled in row-major order
     * @note shape.first = number of rows, shape.second = number of columns
     */
    template<Numeric T>
    std::vector<std::vector<T>> VectorToMatrix(const std::vector<T>& vec, std::pair<int, int> shape);

    /**
     * @brief Checks if computing the volume (product of dimensions) would cause integer overflow.
     *
     * @param shape Tensor shape dimensions
     * @return true if volume computation would exceed INT_MAX
     * 
     * @return false if volume is safe to compute
     *
     * @note Uses long long for intermediate calculations to detect overflow
     * @note Should be called before ShapeToVolume to prevent overflow
     */
    bool IsVolumeOverflow(const std::vector<int>& shape);

    /**
     * @brief Computes the total number of elements (volume) of a tensor shape.
     *
     * @param shape Tensor shape dimensions
     * 
     * @return int Product of all dimensions
     *
     * @throws std::overflow_error if shape is too large (would cause overflow)
     *
     * @note Automatically checks for overflow before computation
     * @note Empty shape returns 1 (scalar)
     */
    int  ShapeToVolume(const std::vector<int>& shape);

    /**
     * @brief Computes stride values for each dimension in row-major order.
     *
     * @param shape Tensor shape dimensions
     * 
     * @return std::vector<int> Stride for each dimension
     *
     * @throws std::overflow_error if shape is too large (would cause overflow)
     *
     * @note Strides are in row-major (C-style) order
     * @note stride[i] = product of all dimensions after dimension i
     * @note Example: shape = {2, 3, 4} ? strides = {12, 4, 1}
     */
    std::vector<int> ShapeToStrides(const std::vector<int>& shape);

    /**
     * @brief Converts multi-dimensional tensor index to flat (1D) array index.
     *
     * @param shape Tensor shape dimensions
     * @param tensor_index Multi-dimensional index coordinates
     * 
     * @return int Corresponding flat index in row-major order
     *
     * @throws std::invalid_argument if size mismatch between shape and tensor_index
     * @throws std::out_of_range if any index value is out of bounds for its dimension
     *
     * @note All indices must be in range [0, shape[i])
     * @note Example: shape = {2, 3}, index = {1, 2} ? flat_index = 5
     */
    int FlatIndex(const std::vector<int>& shape, const std::vector<int>& tensor_index);

    /**
     * @brief Converts flat (1D) array index to multi-dimensional tensor index.
     *
     * @param shape Tensor shape dimensions
     * @param flat_index Flat index in row-major order
     * 
     * @return std::vector<int> Multi-dimensional index coordinates
     *
     * @throws std::invalid_argument if flat_index is out of bounds (>= volume)
     *
     * @note Inverse operation of FlatIndex
     * @note Example: shape = {2, 3}, flat_index = 5 ? tensor_index = {1, 2}
     */
    std::vector<int> TensorIndex(const std::vector<int>& shape, int flat_index);

    /**
     * @brief Checks if two tensor shapes are compatible for broadcasting.
     *
     * @param shape_1 First tensor shape
     * @param shape_2 Second tensor shape
     * 
     * @return true if shapes can be broadcast together
     * @return false if broadcasting is not possible
     *
     * @note Broadcasting rules: dimensions are compatible if they are equal or one is 1
     * @note Comparison starts from the rightmost (trailing) dimensions
     * @note Different number of dimensions is allowed (smaller shape is left-padded with 1s)
     * @note Example: {3, 1, 5} and {1, 4, 5} are compatible ? broadcast to {3, 4, 5}
     */
    bool IsBroadcastCompatible(const std::vector<int>& shape_1, const std::vector<int>& shape_2);

    /**
     * @brief Checks if filter shape is compatible with main tensor for convolution.
     *
     * @param main_shape Shape of the main tensor
     * @param filter_shape Shape of the convolution filter/kernel
     *
     * @return true if filter can be applied to main tensor
     * @return false if dimensions are incompatible
     *
     * @note Filter must not have more dimensions than main tensor
     * @note Each filter dimension must not exceed corresponding main dimension
     * @note Comparison is right-aligned (trailing dimensions)
     */
    bool IsConvolveCompatible(const std::vector<int>& main_shape, const std::vector<int>& filter_shape);

    /**
     * @brief Computes the resulting shape after broadcasting two tensor shapes.
     *
     * @param shape_1 First tensor shape
     * @param shape_2 Second tensor shape
     * 
     * @return std::vector<int> Broadcast result shape
     *
     * @throws std::invalid_argument if shapes are not compatible for broadcasting
     *
     * @note Result dimension = max(dim1, dim2) for each position
     * @note Example: {3, 1, 5} + {1, 4, 5} ? {3, 4, 5}
     */
    std::vector<int> BroadcastShape(const std::vector<int>& shape_1, const std::vector<int>& shape_2);

    /**
     * @brief Computes output shape after applying convolution operation.
     *
     * @param main_shape Shape of the main tensor
     * @param filter_shape Shape of the convolution filter/kernel
     * @param strides Stride values for each dimension
     * 
     * @return std::vector<int> Output feature map shape
     *
     * @throws std::invalid_argument if main_shape and filter_shape are not compatible
     * @throws std::invalid_argument if size mismatch between main_shape and strides
     *
     * @note Formula: output[i] = ((main[i] - filter[i]) / stride[i]) + 1
     * @note Assumes no padding (valid convolution)
     * @note strides must have same dimensionality as main_shape
     */
    std::vector<int> ConvolvedFeatureShape(const std::vector<int>& main_shape, const std::vector<int>& filter_shape, const std::vector<int>& strides);

    /**
     * @brief Finds all integers in a range that are NOT present in the input vector.
     *
     * @param nums Input vector of integers
     * @param bounds Range to search as (lower_inclusive, upper_exclusive)
     * 
     * @return std::vector<int> Sorted vector of missing integers in the range
     *
     * @throws std::invalid_argument if bounds.first >= bounds.second (invalid range)
     *
     * @note Range is [bounds.first, bounds.second)
     * @note Values in nums outside the bounds are ignored
     * @note Result is always sorted in ascending order
     * @note Example: nums = {1, 3, 5}, bounds = {0, 6} ? result = {0, 2, 4}
     */
    std::vector<int> FindRangeComplement(const std::vector<int>& nums, std::pair<int, int> bounds);
}


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
std::vector<std::vector<T>> Utils::TransposeMatrix(const std::vector<std::vector<T>>& matrix)
{
    static_assert(std::is_arithmetic_v<T>, "TransposeMatrix requires a numeric type");

    if (matrix.empty())
    {
        return {};
    }

    size_t rows = matrix.size();
    size_t columns = matrix[0].size();

    if (!Utils::IsRectangular(matrix))
    {
        throw std::invalid_argument("[Tensor-Utils] Transposing Matrix failed: input is not a rectangular matrix.");
    }

    std::vector<std::vector<T>> transposed_matrix(columns, std::vector<T>(rows));

    for (size_t i = 0; i < columns; i++)
    {
        for (size_t j = 0; j < rows; j++)
        {
            transposed_matrix[i][j] = matrix[j][i];
        }
    }

    return transposed_matrix;
}

template<Numeric T>
std::vector<std::vector<T>> Utils::StandardMatrixMultiply(const std::vector<std::vector<T>>& matrix_1, const std::vector<std::vector<T>>& matrix_2)
{
    static_assert(std::is_arithmetic_v<T>, "StandardMatrixMultiply requires a numeric type");

    if (matrix_1.empty() || matrix_2.empty())
    {
        throw std::invalid_argument("[Tensor-Utils] Matrix Multiplication failed: input matrices must not be empty.");
    }

    if (matrix_1[0].size() != matrix_2.size())
    {
        throw std::invalid_argument("[Tensor-Utils] Matrix Multiplication failed: matrix shapes are not compatible for matrix multiplication.");
    }

    size_t rows = matrix_1.size();
    size_t columns = matrix_2[0].size();
    size_t inner = matrix_2.size();

    if (!Utils::IsRectangular(matrix_1))
    {
        throw std::invalid_argument("[Tensor-Utils] Matrix Multiplication failed: first matrix is not rectangular.");
    }

    if (!Utils::IsRectangular(matrix_2))
    {
        throw std::invalid_argument("[Tensor-Utils] Matrix Multiplication failed: second matrix is not rectangular.");
    }

    std::vector<std::vector<T>> result(rows, std::vector<T>(columns, static_cast<T>(0)));

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < columns; j++)
        {
            T sum = static_cast<T>(0);
            for (size_t k = 0; k < inner; k++)
            {
                sum += (matrix_1[i][k] * matrix_2[k][j]);
            }
            result[i][j] = sum;
        }
    }

    return result;
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
