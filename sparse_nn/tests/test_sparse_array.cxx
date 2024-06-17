#include "sparse_array.hxx"
#include "array.hxx"
#include "utils.hxx"
#include <chrono>
#include <omp.h>
#include <random>
#include <vector>

#include <iostream>

int main()
{
    unsigned error = 0;

    const unsigned n_channels = 1, n_rows = 5, n_cols = 3;
    
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> dist_real(0.5, 1.5);
    auto lambda = [&] (){ return dist_real(rng); };

    Array<double> array(n_channels, n_rows, n_cols);  array.fill(lambda);

    SparseArray<double> sparse_array(array);

    for(unsigned element=0; element<sparse_array.size; ++element){
        if ( notZero(sparse_array.Value(element) - array(element)) ) ++error;
    }
    if ( sparse_array.toDense() != array ) ++error;    
    
    std::cout << array << "\n";
    std::cout << sparse_array << std::endl;
    

    Array<double> input_array(n_cols, 1); input_array.fill(lambda);
    Array<double> output_array_strassen = array.mult(input_array);
    Array<double> output_array_sparse = sparse_array.mult(input_array);

    if (output_array_sparse != output_array_strassen) ++error;

    output_array_strassen = array.mult_vec(input_array);
    output_array_sparse = sparse_array.mult_vec(input_array);

    if (output_array_sparse != output_array_strassen) ++error;

    // test move and copy constructors
    SparseArray<double> giver(std::move(sparse_array));
    SparseArray<double> receiver(giver);
    if ( receiver != giver ) ++error;   

    
    // test move and copy assignment
    SparseArray<double> mover(array);
    receiver = mover; // copy
    if ( receiver != mover ) ++error;
  
    receiver = std::move(mover);
    if ( receiver.toDense() != array ) ++error;    

    // test convertion of a vector of arrays
    unsigned n_elements = 2;
    std::vector<Array<double>> vec_array;
    for(unsigned element = 0; element < n_elements; ++element)
    {
        vec_array.push_back(Array<double>(1, 2, 2));
        vec_array.back().fill(double(element));
    }

    std::vector<SparseArray<double>> vec_sparse_array;
    for(unsigned element = 0; element < n_elements; ++element)
    {
        vec_sparse_array.push_back(SparseArray<double>(vec_array[element])); // is r value
        if ( vec_sparse_array.back().toDense() != vec_array[element]) ++error;
    }
    
    // multiplication speed
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;  

    const unsigned n_rows2 = 500, n_cols2 = 300;
    Array<double> mat2(n_rows2, n_cols2); mat2.fill(lambda);
    Array<double> vec2(n_cols2); vec2.fill(1.0);

    SparseArray<double> sparse_mat2(mat2, 1.0); // all the element under 1 in absolute value are ignored


    // parallel sparse multiplication
    begin = std::chrono::steady_clock::now();
    sparse_mat2.mult(vec2);
    end = std::chrono::steady_clock::now();
    double duration_para = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    // comparison with Strassen
    begin = std::chrono::steady_clock::now();
    mat2.mult(vec2);
    end = std::chrono::steady_clock::now();
    double duration_para_strassen = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    if ( duration_para_strassen < duration_para ) ++error;

    // sequential sparse multiplication
    omp_set_num_threads(1);
    begin = std::chrono::steady_clock::now();
    sparse_mat2.mult(vec2);
    end = std::chrono::steady_clock::now();
    double duration_sec = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    begin = std::chrono::steady_clock::now();
    mat2.mult(vec2);
    end = std::chrono::steady_clock::now();
    double duration_sec_strassen = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    if ( duration_sec_strassen < duration_sec ) ++error;
    if ( duration_sec < duration_para ) ++error;

    std::cout << "duration parallelised sparse multiplication: " << duration_para << "\n";
    std::cout << "duration sequential sparse multiplication: " << duration_sec << "\n";
    std::cout << "duration parallelised Strassen multiplication: " << duration_para_strassen << "\n";
    std::cout << "duration sequential Strassen multiplication: " << duration_sec_strassen << "\n";
    std::cout << "proportion of zeros: " << double(sparse_mat2.nb_not_zeros) / sparse_mat2.size << std::endl;
    

    return error;
}