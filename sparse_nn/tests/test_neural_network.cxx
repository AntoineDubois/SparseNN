#include "neural_network.hxx"
#include "layers.hxx"

#include <iostream>
#include <chrono>


inline unsigned out_dim(unsigned input_size, unsigned padding, unsigned stride, unsigned kernel_size)
{
    return (input_size +2*padding -kernel_size)/stride +1;
}

int main()
{
    unsigned error = 0;

    unsigned n_input_channels = 1;
    unsigned n_input_rows = 10;
    unsigned n_input_cols = 5;
    unsigned n_output_channels = 1;
    unsigned kernel_size = 3;
    unsigned stride = 1;
    unsigned padding = 0;

    double kernel_init_value = 1.0;
    double bias_init_value = 0.0;

    Array<double> image(n_input_channels, n_input_rows, n_input_cols); 
    image.fill(1.0);

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;  
    // test dense network
    NeuralNetwork<double> nn;
    Convolution<double>* conv = new Convolution<double>(n_input_channels, n_output_channels, kernel_size, stride, padding,
                    kernel_init_value, bias_init_value);
    unsigned hidden_nrows = out_dim(n_input_rows, padding, stride, kernel_size);
    unsigned hidden_ncols = out_dim(n_input_cols, padding, stride, kernel_size);
    nn.append(conv);
    Max3dPooling<double>* max3d = new Max3dPooling<double>(kernel_size, stride, padding);
    Flatten<double>* flatten = new Flatten<double>();
    hidden_nrows = out_dim(hidden_nrows, padding, stride, kernel_size);
    hidden_ncols = out_dim(hidden_ncols, padding, stride, kernel_size);
    nn.append(max3d, flatten);
   
    
    

    unsigned dense_input_size = hidden_nrows * hidden_ncols;
    std::cout << "dense input size: " << dense_input_size << "\n";
    unsigned output_size = 10;
    Linear<double>* linear = new Linear<double>(dense_input_size, output_size, 1.0, 0.0);
    LogSoftmax<double>* softmax = new LogSoftmax<double>();
    nn.append(linear, softmax);
    
    begin = std::chrono::steady_clock::now();
    Array<double> output = nn(image);
    end = std::chrono::steady_clock::now();
    double duration_dense = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout << output << std::endl;
    
    begin = std::chrono::steady_clock::now();
    output = nn.Forward(image);
    end = std::chrono::steady_clock::now();
    double duration_dense_forward = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    // test sparse network
    SparseConvolution<double>* sparse_conv = new SparseConvolution<double>(*conv);
    SparseLinear<double>* sparse_linear = new SparseLinear<double>(*linear);

    std::cout << nn << std::endl;
    
    nn.layer(0) = sparse_conv;
    delete conv; // eventually, if the layer is not latter used
    nn.layer(2) = sparse_linear;
    delete linear; // eventually, if the layer is not latter used

    begin = std::chrono::steady_clock::now();
    Array<double> output_sparse = nn(image);
    end = std::chrono::steady_clock::now();
    double duration_sparse = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    
    begin = std::chrono::steady_clock::now();
    output_sparse = nn.Forward(image);
    end = std::chrono::steady_clock::now();
    double duration_sparse_forward = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    if (output != output_sparse) ++error;


    std::cout << "Dense:\n";
    std::cout << "\tDuration operator(): " << duration_dense << "\n";
    std::cout << "\tDuration Forward(): " << duration_dense_forward << "\n";
    std::cout << "Sparse:\n";
    std::cout << "\tDuration operator(): " << duration_sparse << "\n";
    std::cout << "\tDuration Forward(): " << duration_sparse_forward << std::endl;


    // additional test
    unsigned n_input_channels2 = 2;
    unsigned n_output_channels2 = 3;
    unsigned nrows2 = 20;
    unsigned ncols2 = 10;
    unsigned kernel_size2 = 3;
    
    NeuralNetwork<double> nn2;
    Convolution<double>* conv2 = new Convolution<double>(n_input_channels2, n_output_channels2, kernel_size2, 1, 0, 1.0, 0.0);
    Flatten<double>* flatten2 = new Flatten<double>();
    unsigned hidden_size = (nrows2 -kernel_size2 +1) * (ncols2 -kernel_size2 +1) * n_output_channels2;
    Linear<double>* linear2 = new Linear<double>(hidden_size, 3, 1.0, 0.0);
    Max<double>* max2 = new Max<double>();

    nn2.append(conv2, flatten2);
    nn2.append(linear2, max2);

    Array<double> input2(n_input_channels2, nrows2, ncols2); 
    input2.fill(1.0);

    Array<double> output2 = nn2(input2);
    std::cout << output2 << std::endl;
    delete conv2; delete flatten2; delete linear2; delete max2;

    // additional test
    auto conv31 = new Convolution<double>(1, 32, 3, 1, 0, 1.0, 0.0);
    auto conv32 = new Convolution<double>(32, 64, 3, 1, 0, 1.0, 0.0);
    auto relu3 = new ReLU<double>();
    auto max_2dpool3 = new Max2dPooling<double>(2, 2, 0);
    auto flatten3 = new Flatten<double>();
    auto linear31 = new Linear<double>(9216, 128, 1.0, 0.0);
    auto linear32 = new Linear<double>(128, 10, 1.0, 0.0);
    auto max3 = new Max<double>();

    NeuralNetwork<double> nn3;
    /*nn3.append(conv31, relu3);
    nn3.append(conv32, relu3);
    nn3.append(max_2dpool3, flatten3);
    nn3.append(linear31, relu3);
    nn3.append(linear32, max3);*/
    nn3.append(conv31);
    nn3.append(conv32);
    nn3.append(max_2dpool3);
    //nn3.append(linear31);
    //nn3.append(linear32);


    Array<double> input3(1, 28, 28); input3.fill(0.1);
    
    
    begin = std::chrono::steady_clock::now();
    nn3.Forward(input3);
    end = std::chrono::steady_clock::now();
    double duration_deep_forward = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    begin = std::chrono::steady_clock::now();
    nn3(input3);
    end = std::chrono::steady_clock::now();
    double duration_deep_inplace = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();


    std::cout << "Deep Network:\n";
    std::cout << "\tDuration operator(): " << duration_deep_inplace << "\n";
    std::cout << "\tDuration Forward(): " << duration_dense_forward << "\n";
    

    return error;
}


