#pragma once

#include <optional>
#include <vector>

namespace LinAlg
{
    class MatrixProperties
    {
    private:
        bool is_square = false;
        bool is_square_synced = false;

        bool is_diagonal = false;
        bool is_diagonal_synced = false;

        bool is_bidiagonal = false;
        bool is_bidiagonal_synced = false;

        bool is_upper_bidiagonal = false;
        bool is_upper_bidiagonal_synced = false;

        bool is_lower_bidiagonal = false;
        bool is_lower_bidiagonal_synced = false;

        bool is_tridiagonal = false;
        bool is_tridiagonal_synced = false;

        bool is_upper_triangular = false;
        bool upper_triangular_synced = false;

        bool is_lower_triangular = false;
        bool lower_triangular_synced = false;

        bool is_symmetric = false;
        bool is_symmetric_synced = false;

        bool is_skew_symmetric = false;
        bool is_skew_symmetric_synced = false;

        bool is_orthogonal = false;
        bool is_orthogonal_synced = false;

        bool is_singular = false;
        bool singular_synced = false;

        bool is_idempotent = false;
        bool is_idempotent_synced = false;

        bool is_nilpotent = false;
        bool is_nilpotent_synced = false;

        bool is_involutory = false;
        bool is_involutory_synced = false;

        std::optional<double> determinant;
        std::optional<int> rank;
        std::optional<double> trace;

        void InvalidateAll();
    };
}
