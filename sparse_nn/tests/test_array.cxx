#include "array.hxx"
#include "utils.hxx"
#include <random>
#include <chrono>

#include <iostream>
#include <omp.h>

int main()
{
    unsigned error = 0;

    //Array<int> array0(1);
    Array<int> tensor1(4, 2, 3); tensor1.fill(2);
    Array<int> matrix1(2, 3);
    Array<int> vector1(3);

    // test dimensions
    if(tensor1.nchannels != 4) ++error;
    if(tensor1.nrows != 2) ++error;
    if(tensor1.ncols != 3) ++error;
    
    // test deterministic matrix and == operator
    if ( tensor1 != 2 ) ++error;
    
    // test random matrix
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> dist_real(0.5, 1.0);
    auto lambda = [&] (){ return dist_real(rng); };
    
    
    Array<double> tensor2(4, 2, 3); 
    tensor2.fill(lambda);

    for(unsigned channel = 0; channel < 4; ++channel){
        for(unsigned row = 0; row < 2; ++row){
            for(unsigned col = 0; col < 3; ++col){
                if (tensor2(channel, row, col) < 0.5 || tensor2(channel, row, col) > 1.0) ++error;
            }
        }
    }
    // test max, argmax, min and argmin
    Array<int> tensor_ext(2, 2, 2);
    tensor_ext(0, 0, 0) = 5;
    tensor_ext(0, 0, 1) = 3;
    tensor_ext(0, 1, 0) = -2;
    tensor_ext(0, 1, 1) = -4;

    tensor_ext(1, 0, 0) = 20;
    tensor_ext(1, 0, 1) = 1;
    tensor_ext(1, 1, 0) = -1;
    tensor_ext(1, 1, 1) = -3;

    int max = tensor_ext.max();
    int min = tensor_ext.min();
    if (max != 20) ++error;
    if (min != -4) ++error;

    unsigned argmax = tensor_ext.argmax();
    unsigned argmin = tensor_ext.argmin();
    if (argmax != 4) ++error;
    if (argmin != 3) ++error;

    // test copy constructor and == operator 
    Array<double> tensor3(tensor2);
    if ( tensor3 != tensor2 ) ++error;
    
    // test reshape
    Array<double> matrix2(2, 3);
    matrix2.fill(lambda);
    Array<double> copymatrix2(matrix2);

    std::cout << "original matrix:" << matrix2 << std::endl;
    matrix2.reshape(3, 2);
    std::cout << "reshaped matrix:" << matrix2 << std::endl;
    for(unsigned element=0; element<6; ++element){
        if( notZero(matrix2(element) -copymatrix2(element)) ) ++error;
    }

    // test flatten
    Array<double> copymatrix3(matrix2);
    matrix2.flatten();
    for(unsigned element=0; element<6; ++element){
        if( notZero(matrix2(element) -copymatrix2(element)) ) ++error;
    }

    // test concatenate channel
    Array<int> matrixA0(2, 3); matrixA0.fill(1);
    Array<int> matrixB0(2, 3); matrixB0.fill(2);

    matrixA0.concatenate(matrixB0, 0);
    std::cout << matrixA0 << std::endl;
    for(unsigned row = 0; row < 2; ++row){
        for(unsigned col = 0; col < 3; ++col){
            if (matrixA0(0, row, col) != 1) ++error;
            if (matrixA0(1, row, col) != 2) ++error;
        }
    }

    // test concatenate row
    Array<int> matrixA1(2, 3); matrixA1.fill(1);
    Array<int> matrixB1(2, 3); matrixB1.fill(2);

    matrixA1.concatenate(matrixB1, 1);
    std::cout << matrixA1 << std::endl;
    for(unsigned row = 0; row < 2; ++row){
        for(unsigned col = 0; col < 3; ++col){
            if (matrixA1(row, col) != 1) ++error;
            if (matrixA1(2 + row, col) != 2) ++error;
        }
    }

    // test concatenate col
    Array<int> matrixA2(2, 3); matrixA2.fill(1);
    Array<int> matrixB2(2, 3); matrixB2.fill(2);

    matrixA2.concatenate(matrixB2, 2);
    std::cout << matrixA2 << std::endl;
    
    for(unsigned row = 0; row < 2; ++row){
        for(unsigned col = 0; col < 3; ++col){
            if (matrixA2(row, col) != 1) ++error;
            if (matrixA2(row, 3 + col) != 2) ++error;
        }
    }
    
    // test flatten
    Array<float> tensor4(5, 2, 1);
    tensor4.flatten();
    if(tensor4.nchannels != 1) ++error;
    if(tensor4.nrows != 10) ++error;
    if(tensor4.ncols != 1) ++error;

    // test copy assignemnt opperator
    Array<int> giver(3); giver.fill(3);
    Array<int> receiver(4); receiver.fill(4);

    receiver = giver;
    if (giver != receiver ) ++error;

    
    // test move assignment opperator
    Array<int> mover(5); mover.fill(5);
    receiver = std::move(mover);
    if (receiver != 5) ++error;
    if (receiver.size != 5) ++error;

    // test mult and linear (if linear is true, mult is too)
    Array<int> matrixA3(3, 2); matrixA3.fill(1);
    Array<int> matrixB3(2, 2); matrixB3.fill(2);
    Array<int> bias3(3); bias3.fill(5);

    Array<int> output3 = matrixA3.linear(matrixB3, bias3);
    if (output3 != 9) ++error;
    

    Array<int> vector_input3(2); vector_input3.fill(2);
    output3 = matrixA3.linear_vec(vector_input3, bias3);
    if (output3 != 9) ++error;

    // comparison parallelised vs sequential
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;  
    
    const unsigned n_rows2 = 500, n_cols2 = 300;
    Array<double> mat2(n_rows2, n_cols2); mat2.fill(lambda);
    Array<double> mat_input_2(n_cols2, 1); mat_input_2.fill(1.0);
    Array<double> vec_input_2(n_cols2); vec_input_2.fill(1.0);

    begin = std::chrono::steady_clock::now();
    mat2.mult(mat_input_2);
    end = std::chrono::steady_clock::now();
    double duration_para_mult = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    begin = std::chrono::steady_clock::now();
    mat2.mult_vec(vec_input_2);
    end = std::chrono::steady_clock::now();
    double duration_para_mult_vec = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    omp_set_num_threads(1);
    begin = std::chrono::steady_clock::now();
    mat2.mult(mat_input_2);
    end = std::chrono::steady_clock::now();
    double duration_sec_mult = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    begin = std::chrono::steady_clock::now();
    mat2.mult_vec(vec_input_2);
    end = std::chrono::steady_clock::now();
    double duration_sec_mult_vec = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    if ( duration_sec_mult < duration_para_mult ) ++error;
    if ( duration_sec_mult_vec < duration_para_mult_vec ) ++error;

    std::cout << "duration parallelised Strassen multiplication: " << duration_para_mult << "\n";
    std::cout << "duration sequential Strassen multiplication: " << duration_sec_mult << "\n";
    
    std::cout << "duration parallelised Strassen multiplication _vec: " << duration_para_mult_vec << "\n";
    std::cout << "duration sequential Strassen multiplication _vec: " << duration_sec_mult_vec << std::endl;
    

    

    return error;
}