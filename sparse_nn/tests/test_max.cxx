#include "algebra.hxx"
#include "./activation_layers/max.hxx"

#include <chrono>
#include <random>
#include <iostream>

int main()
{
    unsigned error = 0;

    Max<int> max_int;
    Array<int> input_array(4);
    input_array(0) = 1;
    input_array(1) = 3;
    input_array(2) = 4;
    input_array(3) = 5;
    
    
    Array<int> output_array = max_int.Forward(input_array);
    if ( output_array.size != 1 ) ++error;
    if ( output_array(0) != 5 ) ++error;

    std::cout << output_array << "\n";
    std::cout << input_array << "\n";

    Array<int> output_array_opp = max_int(input_array);
    if ( output_array_opp.size != 1 ) ++error;
    if ( output_array_opp(0) != 5 ) ++error;

    std::cout << output_array_opp << "\n";
    std::cout << input_array << "\n";


    // inline vs copy speed
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;  

    const unsigned size = 5'000;

    Max<double> max_double;
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> dist_real(0.5, 1.0);
    auto lambda_real = [&] (){ return dist_real(rng); };
    Array<double> array2(size); array2.fill(lambda_real);

    begin = std::chrono::steady_clock::now();
    max_double.Forward(array2);
    end = std::chrono::steady_clock::now();
    double duration_copy = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    // operator() alters the input array to a 1 element array. Therefore, the copy should be executed first.
    begin = std::chrono::steady_clock::now();
    max_double(array2);
    end = std::chrono::steady_clock::now();
    double duration_inplace = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    

    std::cout << "duration_inplace=" << duration_inplace << "\n";
    std::cout << "duration_copy=" << duration_copy << std::endl;


  

    return error;
}