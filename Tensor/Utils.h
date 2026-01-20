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

    template<Numeric T>
    double Norm(const std::vector<T>& nums);

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

#include "Utils.inl"