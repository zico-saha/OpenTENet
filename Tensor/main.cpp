#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#include "Tensor.h"
#include "Initializer.h"
#include "Math.h"
#include "Activation.h"
#include "Matrix.h"

using namespace std;
using namespace std::chrono;

int main()
{
    try
    {
        auto start = high_resolution_clock::now();
        {
            LinAlg::Matrix A({ 4, 3 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
            LinAlg::SVDResult result = A.SVDecomposition();

            cout << "original:\n";
            A.Print();

            cout << "matrix-U:\n";
            result.U.Print();
            cout << "matrix-S:\n";
            result.S.Print();
            cout << "matrix-V:\n";
            result.V.Print();

            LinAlg::Matrix R = result.U.MatMul(result.S.MatMul(result.V.Transpose()));

            cout << "reconstructed:\n";
            R.Print();
        }
        auto end = high_resolution_clock::now();
        duration<double> elapsed = end - start;

        std::cout << "\nExecution time: " << elapsed.count() << " seconds\n";
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << "\n";
    }

    return 0;
}

// OUTPUT:

//original:
//1       2       3
//4       5       6
//7       8       9
//10      11      12
//matrix - U :
//-0.836545 - 0.392045 - 0.013902 - 0.382492
//- 0.473585       0.237634 - 0.275168       0.802203
//- 0.110625       0.700868 - 0.536435 - 0.456929
//0.252335 - 0.546457 - 0.797701       0.0372184
//matrix - S :
//0       0 - 25.4624
//1.29066 0       0
//0       0       0
//0       0       0
//matrix - V :
//0.760776 - 0.408248       0.504533
//0.0571405       0.816497        0.574516
//- 0.646495 - 0.408248       0.644498
//reconstructed :
//10.3618 12.2085 14.0552
//6.3173  6.94538 7.57347
//2.10934 1.66997 1.2306
//- 3.77822 - 3.7316 - 3.68497
//
//Execution time : 0.0173598 seconds

//original:
//1       2       3
//4       5       6
//7       8       9
//10      11      12
//matrix - U :
//-0.0776151 - 0.833052 - 0.392045 - 0.382492
//- 0.31046 - 0.451237       0.237634        0.802203
//- 0.543305 - 0.069421       0.700868 - 0.456929
//- 0.776151       0.312395 - 0.546457       0.0372184
//matrix - S :
//-12.8841        21.8764 0
//0       2.24624 0.613281
//0       0       0
//0       0       0
//matrix - V :
//1       0       0
//0 - 0.667002       0.745056
//0 - 0.745056 - 0.667002
//reconstructed :
//1       2       3
//4       5       6
//7       8       9
//10      11      12
//
//Execution time: 0.013042 seconds