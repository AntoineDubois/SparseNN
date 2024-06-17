#include "algebra.hxx"
#include "utils.hxx"
#include "./activation_layers/softmax.hxx"

#include <chrono>
#include <random>
#include <iostream>
#include <omp.h>

int main()
{
    unsigned error = 0;

    Softmax<double> softmax;
    Array<double> input_array(2); input_array.fill(0.0);
    
    Array<double> output_array = softmax.Forward(input_array);

    if (notZero(output_array(0) -0.5) ) ++error;
    if (notZero(output_array(1) -0.5) ) ++error;
    if (notZero(output_array(0) +output_array(1) -1.0)) ++error;
    
    softmax(input_array);
    if (notZero(input_array(0) -0.5) ) ++error;
    if (notZero(input_array(1) -0.5) ) ++error;
    if (notZero(input_array(0) +input_array(1) -1.0)) ++error;


    // parallel speed
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;  

    const unsigned n_rows = 10, n_cols = 5'000;

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> dist_real(0.5, 1.0);
    auto lambda_real = [&] (){ return dist_real(rng); };
    Array<double> array2(n_rows, n_cols); array2.fill(lambda_real);

    // parallel
    begin = std::chrono::steady_clock::now();
    softmax(array2);
    end = std::chrono::steady_clock::now();
    double duration_para_inplace = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    begin = std::chrono::steady_clock::now();
    softmax.Forward(array2);
    end = std::chrono::steady_clock::now();
    double duration_para_copy = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
  
    // sequential
    omp_set_num_threads(1);
    
    begin = std::chrono::steady_clock::now();
    softmax(array2);
    end = std::chrono::steady_clock::now();
    double duration_sec_inplace = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    begin = std::chrono::steady_clock::now();
    softmax.Forward(array2);
    end = std::chrono::steady_clock::now();
    double duration_sec_copy = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    if ( duration_sec_copy < duration_para_copy ) ++error;
    if ( duration_sec_inplace < duration_para_inplace ) ++error;

    std::cout << "duration_sec_copy=" << duration_sec_copy << "\n";
    std::cout << "duration_para_copy=" << duration_para_copy << "\n";
    std::cout << "duration_sec_inplace=" << duration_sec_inplace << "\n";
    std::cout << "duration_para_inplace=" << duration_para_inplace << std::endl;

  

    return error;
}