#include <iostream>



int main(void) {
    unsigned long long myvar = 0;
    myvar |= (int) (2 > 1);
    for (int k = 0; k < 10; k++){
        myvar <<= 1;
    }

    std::cout << "myvar: " << myvar << std::endl;
    return 0;
}