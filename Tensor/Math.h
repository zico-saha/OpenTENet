#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

// Forward declarations
class Tensor;
class Matrix;

// ========================================
// Concept: Only Tensor or Matrix allowed
// ========================================
template <typename T>
concept MathContainer = std::same_as<T, Tensor> || std::same_as<T, Matrix>;

// ========================================
// Math Class
// ========================================
class Math
{
private:
    static constexpr double EXP_BASE_LIMIT = 700.0;
    static constexpr double EPSILON_SCALE = 1e6;

public:
    // ========================================
    // Elementary Methods
    // ========================================
    template <MathContainer T>
    static T Abs(const T& x);

    template <MathContainer T>
    static T Ceil(const T& x);

    template <MathContainer T>
    static T Clip(const T& x, double min_value, double max_value);

    template <MathContainer T>
    static T Exp(const T& x);

    template <MathContainer T>
    static T Floor(const T& x);

    template <MathContainer T>
    static T Log(const T& x, double base = std::numbers::e);

    template <MathContainer T>
    static T Mod(const T& x, double mod_value);

    template <MathContainer T>
    static T Power(const T& x, double exponent);

    template <MathContainer T>
    static T Round(const T& x, int decimal_place = 2);

    template <MathContainer T>
    static T Sqrt(const T& x);

    // ========================================
    // Trigonometric Methods
    // ========================================
    template <MathContainer T>
    static T Sin(const T& x);

    template <MathContainer T>
    static T Cos(const T& x);

    template <MathContainer T>
    static T Tan(const T& x);

    template <MathContainer T>
    static T Csc(const T& x);

    template <MathContainer T>
    static T Sec(const T& x);

    template <MathContainer T>
    static T Cot(const T& x);

    // ========================================
    // Inverse Trigonometric Methods
    // ========================================
    template <MathContainer T>
    static T Asin(const T& x);

    template <MathContainer T>
    static T Acos(const T& x);

    template <MathContainer T>
    static T Atan(const T& x);

    template <MathContainer T>
    static T Acsc(const T& x);

    template <MathContainer T>
    static T Asec(const T& x);

    template <MathContainer T>
    static T Acot(const T& x);

    // ========================================
    // Hyperbolic Methods
    // ========================================
    template <MathContainer T>
    static T Sinh(const T& x);

    template <MathContainer T>
    static T Cosh(const T& x);

    template <MathContainer T>
    static T Tanh(const T& x);

    template <MathContainer T>
    static T Csch(const T& x);

    template <MathContainer T>
    static T Sech(const T& x);

    template <MathContainer T>
    static T Coth(const T& x);

    // ========================================
    // Inverse Hyperbolic Methods
    // ========================================
    template <MathContainer T>
    static T Asinh(const T& x);

    template <MathContainer T>
    static T Acosh(const T& x);

    template <MathContainer T>
    static T Atanh(const T& x);

    template <MathContainer T>
    static T Acsch(const T& x);

    template <MathContainer T>
    static T Asech(const T& x);

    template <MathContainer T>
    static T Acoth(const T& x);
};

#include "Math.inl"