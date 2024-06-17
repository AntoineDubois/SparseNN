#include "algebra.hxx"
#include "./layers/max_3d_pooling.hxx"


#include <random>
#include <chrono>
#include <omp.h>

int main()
{
    unsigned error = 0;

    unsigned n_input_channels = 2;
    unsigned n_input_rows = 10;
    unsigned n_input_cols = 5;
    unsigned n_output_channels = 1;
    unsigned kernel_size = 3;
    unsigned stride = 1;
    unsigned padding = 0;
    

    double kernel_init_value = 1.0, bias_init_value = 2.0;

    Array<double> image(n_input_channels, n_input_rows, n_input_cols); 
    image.fill(1.0);
    
    Max3dPooling<double> max_pooling_layer(kernel_size, stride, padding);
    
    
    Array<double> output_image = max_pooling_layer(image);
    Array<double> output_image_forward = max_pooling_layer.Forward(image);
    
    
    // check output dimensions
    if ( output_image.nchannels != n_output_channels) ++error;
    if ( output_image.nrows != 8) ++error;
    if ( output_image.ncols != 3) ++error;

    if ( output_image_forward.nchannels != n_output_channels) ++error;
    if ( output_image_forward.nrows != 8) ++error;
    if ( output_image_forward.ncols != 3) ++error;


    double true_value = 1.0;
    std::cout << "true value:" << true_value << std::endl; // only true for constant kernel and constant bias values, and no padding

    std::cout << output_image << std::endl;
    std::cout << output_image_forward << std::endl;

    if ( output_image != true_value ) ++error;
    if ( output_image_forward != true_value) ++error;

    // test stride and padding
    stride = 2;
    padding = 1;
    kernel_size = 5;

    Max3dPooling<double> max_pooling_layer_padding_stride(kernel_size, stride, padding);    
    Array<double> output_image_forward2 = max_pooling_layer_padding_stride.Forward(image);// = symbol is not defined
    //std::cout << output_image_forward2 << std::endl;

    
    // parallel speed
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;  

    unsigned n_input_channels2 = 3;
    unsigned n_input_rows2 = 100;
    unsigned n_input_cols2 = 50;
    unsigned n_output_channels2 = 60;
    unsigned kernel_size2 = 3;
    unsigned stride2 = 1;
    unsigned padding2 = 0;

    double kernel_init_value2 = 1.0, bias_init_value2 = 2.0;

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> dist_real(0.5, 1.0);
    auto lambda_real = [&] (){ return dist_real(rng); };
    
    Array<double> image2(n_input_channels2, n_input_rows2, n_input_cols2); 
    image2.fill(lambda_real);
    
    Max3dPooling<double> max_pooling_layer2(kernel_size2, stride2, padding2);

    // parallel
    begin = std::chrono::steady_clock::now();
    max_pooling_layer2(image2);
    end = std::chrono::steady_clock::now();
    double duration_para = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    begin = std::chrono::steady_clock::now();
    max_pooling_layer2.Forward(image2);
    end = std::chrono::steady_clock::now();
    double duration_para_forward = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
  
    // sequential
    omp_set_num_threads(1);
    begin = std::chrono::steady_clock::now();
    max_pooling_layer2(image2);
    end = std::chrono::steady_clock::now();
    double duration_sec = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    begin = std::chrono::steady_clock::now();
    max_pooling_layer2.Forward(image2);
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