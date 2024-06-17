#include "algebra.hxx"
#include "./activation_layers/flatten.hxx"

#include <chrono>
#include <random>
#include <iostream>

int main()
{
    unsigned error = 0;

    Flatten<int> flatten_int;
    Array<int> input_array(3, 2);
    input_array(0, 0) = 1;
    input_array(0, 1) = 2;
    input_array(1, 0) = 3;
    input_array(1, 1) = 4;
    input_array(2, 0) = 5;
    input_array(2, 1) = 6;
    
    
    Array<int> output_array = flatten_int.Forward(input_array);
    if (notZero(output_array(0) -1) ) ++error;
    if (notZero(output_array(1) -2) ) ++error;
    if (notZero(output_array(2) -3) ) ++error;
    if (notZero(output_array(3) -4) ) ++error;
    if (notZero(output_array(4) -5) ) ++error;
    if (notZero(output_array(5) -6) ) ++error;

    std::cout << output_array << "\n";
    std::cout << input_array << "\n";

    Array<int> output_array_opp = flatten_int(input_array);
    if (notZero(input_array(0) -1) ) ++error;
    if (notZero(input_array(1) -2) ) ++error;
    if (notZero(input_array(2) -3) ) ++error;
    if (notZero(input_array(3) -4) ) ++error;
    if (notZero(input_array(4) -5) ) ++error;
    if (notZero(input_array(5) -6) ) ++error;

    std::cout << output_array_opp << "\n";
    std::cout << input_array << "\n";


    // inline vs copy speed
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;  

    const unsigned n_rows = 100, n_cols = 5'000;

    Flatten<double> flatten_double;
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> dist_real(0.5, 1.0);
    auto lambda_real = [&] (){ return dist_real(rng); };
    Array<double> array2(n_rows, n_cols); array2.fill(lambda_real);

    // copy
    begin = std::chrono::steady_clock::now();
    Array<double> out_copy( flatten_double.Forward(array2));
    end = std::chrono::steady_clock::now();
    double duration_copy = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    // inplace 
    begin = std::chrono::steady_clock::now();
    Array<double> out_inplace = std::move(flatten_double(array2));
    end = std::chrono::steady_clock::now();
    double duration_inplace = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout << array2.ncols << std::endl;
    
    std::cout << "error: " << error << std::endl;
    if ( duration_copy < duration_inplace ) ++error;

    std::cout << "duration_inplace=" << duration_inplace << "\n";
    std::cout << "duration_copy=" << duration_copy << std::endl;

  

    return error;
}