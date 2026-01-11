#include <iostream>
#include <vector>
#include <cmath>

#include "Tensor.h"
#include "Initializer.h"
#include "Math.h"
#include "Activation.h"
#include "Matrix.h"

using namespace std;

int main()
{
    try
    {
        LinAlg::Matrix A({ {2, 1, 5, 0}, {-2, 1, 4, -1}, {3, 0, 1, -1} });
        
        LinAlg::QRResult r = A.QRDecomposition();

        cout << "===== QR RESULT =====\n";
        cout << "Matrix-P:\n";
        r.Q.Print();
        cout << "Matrix-L:\n";
        r.R.Print();
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << "\n";
    }

    return 0;
}

//==== = QR RESULT ==== =
//Matrix - P:
//0.707107 0.408248 - 0.57735
//0.707107 - 0.408248 0.57735
//0 0.816497 0.57735
//Matrix - L :
//1.41421 0.707107 0.707107
//0 1.22474 0.408248
//0 0 1.1547