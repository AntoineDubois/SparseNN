#include "algebra.hxx"
#include "./layers/linear.hxx"
#include "./layers/sparse_linear.hxx"

#include <omp.h>
#include <chrono>
#include <random>
#include <iostream>

int main()
{
    unsigned error = 0;

    unsigned input_size = 2, output_size = 3, batch_size = 3;
    Linear<double> linear_layer(input_size, output_size, 1.0, 2.0);
    SparseLinear<double> sparse_linear_layer(linear_layer);

    Array<double> input_vector(input_size, batch_size); input_vector.fill(1.0);
    Array<double> output = sparse_linear_layer(input_vector);
    if ( output != 4.0 ) ++error;

    Array<double> output_forward = sparse_linear_layer.Forward(input_vector);
    if ( output_forward != 4.0 ) ++error;
    
    // parallel speed
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;  

    const unsigned input_size2 = 100, batch_size2 = 500, output_size2 = 50;

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> dist_real(0.5, 1.0);
    auto lambda_real = [&] (){ return dist_real(rng); };
    Array<double> array2(input_size2, batch_size2); array2.fill(lambda_real);
    Linear<double> linear_layer2(input_size2, output_size2, 1.0, 2.0);
    SparseLinear<double> sparse_linear_layer2(linear_layer2);

    // parallel
    begin = std::chrono::steady_clock::now();
    sparse_linear_layer2(array2);
    end = std::chrono::steady_clock::now();
    double duration_para_sparse = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    begin = std::chrono::steady_clock::now();
    sparse_linear_layer2.Forward(array2);
    end = std::chrono::steady_clock::now();
    double duration_para_sparse_forward = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    
    begin = std::chrono::steady_clock::now();
    linear_layer2(array2);
    end = std::chrono::steady_clock::now();
    double duration_para_dense = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    begin = std::chrono::steady_clock::now();
    linear_layer2.Forward(array2);
    end = std::chrono::steady_clock::now();
    double duration_para_dense_forward = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
  
    // sequential
    omp_set_num_threads(1);
    
    begin = std::chrono::steady_clock::now();
    sparse_linear_layer2(array2);
    end = std::chrono::steady_clock::now();
    double duration_sec_sparse = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    begin = std::chrono::steady_clock::now();
    sparse_linear_layer2.Forward(array2);
    end = std::chrono::steady_clock::now();
    double duration_sec_sparse_forward = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    begin = std::chrono::steady_clock::now();
    linear_layer2(array2);
    end = std::chrono::steady_clock::now();
    double duration_sec_dense = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    begin = std::chrono::steady_clock::now();
    linear_layer2.Forward(array2);
    end = std::chrono::steady_clock::now();
    double duration_sec_dense_forward = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
  

    if ( duration_sec_sparse < duration_para_sparse ) ++error;
    if ( duration_sec_sparse_forward < duration_para_sparse_forward ) ++error;

    std::cout << "Sparse:\n";
    std::cout << "\tduration_sec=" << duration_sec_sparse << "\n";
    std::cout << "\tduration_para=" << duration_para_sparse << "\n";
    std::cout << "\tduration_sec_forward=" << duration_sec_sparse_forward << "\n";
    std::cout << "\tduration_para_forward=" << duration_para_sparse_forward << "\n";

    std::cout << "Dense:\n";
    std::cout << "\tduration_sec=" << duration_sec_dense << "\n";
    std::cout << "\tduration_para=" << duration_para_dense << "\n";
    std::cout << "\tduration_sec_forward=" << duration_sec_dense_forward << "\n";
    std::cout << "\tduration_para_forward=" << duration_para_dense_forward << std::endl;

    return error;
}