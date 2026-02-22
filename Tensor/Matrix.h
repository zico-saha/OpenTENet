#pragma once

#include "Utils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class Math;

namespace LinAlg
{
    struct EliminationResult;
    struct LUResult;
    struct LDUResult;
    struct QRResult;
    struct SVDResult;
    struct CholeskyResult;
    struct EigenResult;
    struct GKBResult;

    class Matrix
    {
        friend class Math;

    private:
        std::vector<std::vector<double>> data;

        std::vector<std::pair<double, int>> sparse_data;

        std::pair<int, int> shape = { 0, 0 };

        int volume = 0;

        // ========== Constants ==========
        static constexpr double TOLERANCE = 1e-9;

    private:
        bool IsFullColumnRank() const;

        bool IsFullRowRank() const;

        void ClearNoise();

        LinAlg::Matrix Apply(const std::function<double(double)>& _func) const;

        std::pair<double, double> ComputeGivens(const double& _value_1, const double& _value_2) const;

        double WilkinsonShift() const;

        Matrix PartialMatMul(const Matrix& _sub_matrix, const std::pair<int, int>& _start, const std::pair<int, int>& _end, const bool& _left_multiply = false) const;

        void PermuteRows(const std::vector<int>& _permutation);

        void PermuteColumns(const std::vector<int>& _permutation);

    public:
        Matrix() {}

        Matrix(const std::pair<int, int>& _shape, const double& _value);

        Matrix(const std::pair<int, int>& _shape, const std::vector<double>& _data);

        static Matrix Identity(const int& _n, const double& _scale = 1.0);

        static Matrix RandomUniform(const int& _rows, const int& _cols, const double& _min_value = -1.0, const double& _max_value = 1.0, std::optional<unsigned int> seed = std::nullopt);

        static Matrix RandomNormal(const int& _rows, const int& _cols, const double& _mean = 0.0, const double& _std_dev = 1.0, std::optional<unsigned int> seed = std::nullopt);
        
        static Matrix Diagonal(const std::vector<double>& _diag_values);

        std::pair<int, int> Shape() const;

        int Row() const;

        int Column() const;

        int Volume() const;

        bool IsEmpty() const;

        bool IsSquare() const;

        bool IsDiagonal(const double& _tolerance = Matrix::TOLERANCE) const;

        bool IsBidiagonal(const std::string& _type = "any", const double& _tolerance = Matrix::TOLERANCE) const;

        bool IsTridiagonal(const double& _tolerance = Matrix::TOLERANCE) const;

        bool IsUpperTriangular(const double& _tolerance = Matrix::TOLERANCE) const;

        bool IsLowerTriangular(const double& _tolerance = Matrix::TOLERANCE) const;

        bool IsSymmetric(const double& _tolerance = Matrix::TOLERANCE) const;

        bool IsSkewSymmetric(const double& _tolerance = Matrix::TOLERANCE) const;

        bool IsOrthogonal() const;

        bool IsSingular(const double& _tolerance = Matrix::TOLERANCE) const;

        bool IsIdempotent() const;

        bool IsNilpotent(const int& max_power = 10, const double& _tolerance = Matrix::TOLERANCE) const; // Implement later

        bool IsInvolutory(const double& _tolerance = Matrix::TOLERANCE) const; // Implement later

        void operator=(const Matrix& _matrix);

        bool operator==(const Matrix& _matrix) const;

        bool operator!=(const Matrix& _matrix) const;

        Matrix operator+(const double& _scalar) const;

        Matrix operator-(const double& _scalar) const;

        Matrix operator*(const double& _scalar) const;

        Matrix operator/(const double& _scalar) const;

        Matrix operator+(const std::vector<double>& _vector) const;

        Matrix operator-(const std::vector<double>& _vector) const;

        Matrix operator*(const std::vector<double>& _vector) const;

        Matrix operator/(const std::vector<double>& _vector) const;

        Matrix operator+(const Matrix& _matrix) const;

        Matrix operator-(const Matrix& _matrix) const;

        Matrix operator*(const Matrix& _matrix) const;

        Matrix operator/(const Matrix& _matrix) const;

        void operator+=(const double& _scalar);

        void operator-=(const double& _scalar);

        void operator*=(const double& _scalar);

        void operator/=(const double& _scalar);

        void operator+=(const std::vector<double>& _vector);

        void operator-=(const std::vector<double>& _vector);

        void operator*=(const std::vector<double>& _vector);

        void operator/=(const std::vector<double>& _vector);

        void operator+=(const Matrix& _matrix);

        void operator-=(const Matrix& _matrix);

        void operator*=(const Matrix& _matrix);

        void operator/=(const Matrix& _matrix);

        Matrix AddColumnwise(const std::vector<double>& _vector) const;

        Matrix SubtractColumnwise(const std::vector<double>& _vector) const;

        Matrix MultiplyColumnwise(const std::vector<double>& _vector) const;

        Matrix DivideColumnwise(const std::vector<double>& _vector) const;

        Matrix MatMul(const std::vector<std::vector<double>>& _matrix) const;

        Matrix MatMul(const Matrix& _matrix) const;

        static Matrix MatMul(const Matrix& _matrix_1, const Matrix& _matrix_2);

        Matrix Transpose() const;

        Matrix Inverse() const;

        Matrix PseudoInverse() const;  // Moore-Penrose: Modify later after SVD

        LinAlg::EliminationResult GaussianElimination(const Matrix& _aug_matrix = Matrix()) const;

        LinAlg::EliminationResult GaussJordanElimination(const Matrix& _aug_matrix = Matrix()) const;

        double Determinant() const;

        double Trace() const;

        int Rank() const;

        std::vector<double> Diag(const bool& _sign = false) const;

        Matrix ReduceSum(const bool& _row_wise = true) const;

        Matrix ReduceMean(const bool& _row_wise = true) const;

        Matrix ReduceVar(const bool& _row_wise = true, const bool& _inference = false) const;

        Matrix ReduceMax(const bool& _row_wise = true) const;

        Matrix ReduceMin(const bool& _row_wise = true) const;

        double Sum() const;

        double Mean() const;

        double Var(const bool& _inference = false) const;

        double Max()const;

        double Min() const;

        Matrix Reshape(const std::pair<int, int> _shape) const;

        void SwapRows(const int& _row_1, const int& _row_2);

        void SwapColumns(const int& _col_1, const int& _col_2);

        void Patch(const Matrix& _matrix, const std::pair<int, int>& _start, const std::pair<int, int>& _end);

        Matrix Submatrix(const std::pair<int, int>& _start, const std::pair<int, int>& _end) const;

        std::vector<double> GetRow(const int& _row_index) const;

        std::vector<double> GetColumn(const int& _column_index) const;

        Matrix GetRows(const std::vector<int>& _row_indices) const;

        Matrix GetColumns(const std::vector<int>& _column_indices) const;

        std::vector<double> GetFlatData() const;

        void PushRow(const std::vector<double>& _row_data);

        void PushColumn(const std::vector<double>& _column_data);

        void PopRow(const int& _index = -1);

        void PopColumn(const int& _index = -1);

        void PopRows(const std::vector<int>& _indices);

        void PopColumns(const std::vector<int>& _indices);

        double FrobeniusNorm() const;

        double SpectralNorm() const;

        double NuclearNorm() const;

        double InfinityNorm() const;

        double OneNorm() const;

        LinAlg::LUResult LUDecomposition() const;

        LinAlg::LDUResult LDUDecomposition() const;

        LinAlg::QRResult GSQRDecomposition() const;

        LinAlg::QRResult HQRDecomposition(const bool& _full = true) const;

        LinAlg::SVDResult SVDecomposition() const;

        LinAlg::CholeskyResult CholeskyDecomposition() const;

        LinAlg::EigenResult EigenDecomposition() const;

        LinAlg::EigenResult SpectralDecomposition() const;  // For symmetric matrices

        LinAlg::GKBResult GKBidiagonalize() const;

        LinAlg::SVDResult GRDiagonalize() const;

        void Print() const;
    };
}
