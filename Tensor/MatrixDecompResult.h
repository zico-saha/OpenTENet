#pragma

#include "Matrix.h"

#include <vector>

namespace LinAlg
{
    struct EliminationResult
    {
        Matrix A;
        Matrix B;
        int rank = 0;
        int swapCount = 0;
    };

    struct LUResult
    {
        Matrix L;
        Matrix U;
        Matrix P;
    };

    struct LDUResult
    {
        Matrix L;
        Matrix D;
        Matrix U;
        Matrix P;
    };

    struct QRResult
    {
        Matrix Q;
        Matrix R;
    };

    struct GKBResult
    {
        Matrix U;
        Matrix B;
        Matrix V;
    };

    struct SVDResult
    {
        Matrix U;
        Matrix S;
        Matrix V;
    };

    struct EigenResult
    {
        std::vector<double> eigenvalues;
        Matrix eigenvectors;
    };

    struct CholeskyResult
    {
        Matrix L;
    };
}
