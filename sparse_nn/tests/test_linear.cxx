#include "algebra.hxx"
#include "./layers/linear.hxx"

#include <omp.h>
#include <chrono>
#include <random>
#include <iostream>

int main()
{
    unsigned error = 0;

    unsigned input_size = 2, output_size = 3, batch_size = 3;
    Linear<double> linear_layer(input_size, output_size, 1.0, 2.0);
    
    Array<double> input_vector(input_size, batch_size); input_vector.fill(1.0);
    Array<double> output = linear_layer(input_vector);
    if ( output != 4.0 ) ++error;

    Array<double> output_forward = linear_layer.Forward(input_vector);
    if ( output_forward != 4.0 ) ++error;
    
    // parallel speed
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;  

    const unsigned input_size2 = 10, batch_size2 = 500, output_size2 = 5;

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> dist_real(0.5, 1.0);
    auto lambda_real = [&] (){ return dist_real(rng); };
    Array<double> array2(input_size2, batch_size2); array2.fill(lambda_real);
    Linear<double> linear_layer2(input_size2, output_size2, 1.0, 2.0);

    // parallel
    begin = std::chrono::steady_clock::now();
    linear_layer2(array2);
    end = std::chrono::steady_clock::now();
    double duration_para = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    begin = std::chrono::steady_clock::now();
    linear_layer2.Forward(array2);
    end = std::chrono::steady_clock::now();
    double duration_para_forward = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
  
    // sequential
    omp_set_num_threads(1);
    
    begin = std::chrono::steady_clock::now();
    linear_layer2(array2);
    end = std::chrono::steady_clock::now();
    double duration_sec = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    begin = std::chrono::steady_clock::now();
    linear_layer2.Forward(array2);
    end = std::chrono::steady_clock::now();
    double duration_sec_forward = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    if ( duration_sec < duration_para ) ++error;
    if ( duration_sec_forward < duration_para_forward ) ++error;

    std::cout << "duration_sec=" << duration_sec << "\n";
    std::cout << "duration_para=" << duration_para << "\n";
    std::cout << "duration_sec_forward=" << duration_sec_forward << "\n";
    std::cout << "duration_para_forward=" << duration_para_forward << std::endl;

    return error;
}