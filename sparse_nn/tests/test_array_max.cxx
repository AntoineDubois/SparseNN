#include "array.hxx"

#include <random>
#include <chrono>
#include <iostream>
#include <omp.h>

int main()
{
    unsigned error = 0;

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> dist_real(0.5, 1.0);
    auto lambda = [&] (){ return dist_real(rng); };
    
    // inline vs copy max and min speed
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;  

    //const unsigned size = 1'000'000;
    const unsigned size = 100'000;
    Array<double> tensor(size); 
    tensor.fill(lambda);

    begin = std::chrono::steady_clock::now();
    std::cout << tensor.max() << std::endl;
    end = std::chrono::steady_clock::now();
    double duration_max = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    begin = std::chrono::steady_clock::now();
    tensor.min();
    end = std::chrono::steady_clock::now();
    double duration_min = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    
    begin = std::chrono::steady_clock::now();
    tensor.argmax();
    end = std::chrono::steady_clock::now();
    double duration_arg_max = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    begin = std::chrono::steady_clock::now();
    tensor.argmin();
    end = std::chrono::steady_clock::now();
    double duration_arg_min = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();


    std::cout << "duration Max: " << duration_max << "\n";
    std::cout << "duration arg Max: " << duration_arg_max << "\n";
    
    std::cout << "duration Min: " << duration_min << "\n";
    std::cout << "duration arg Min: " << duration_arg_min << std::endl;
    

    return error;
}