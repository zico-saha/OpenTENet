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
        Tensor T({ 2, 2 }, { 1.6754332, 0.0587, -2.616641, 0.677100 });
        
        Tensor result = Math::Round(T, 1);
        result.Print();
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << "\n";
    }

    return 0;
}
