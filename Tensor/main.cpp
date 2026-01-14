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
        LinAlg::Matrix A({ {2, -1, -2}, {-4, 6, 3}, {-4, -2, 8} });
        
        LinAlg::QRResult r = A.HQRDecomposition();

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
