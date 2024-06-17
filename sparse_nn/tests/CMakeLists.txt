include_directories("../include/algebra")

add_executable(test_utils test_utils.cxx)
add_executable(test_array test_array.cxx)
add_executable(test_array_max test_array_max.cxx)
add_executable(test_sparse_array test_sparse_array.cxx)

include_directories("../include/network")
add_executable(test_flatten test_flatten.cxx)
add_executable(test_max test_max.cxx)
add_executable(test_logsoftmax test_logsoftmax.cxx)
add_executable(test_relu test_relu.cxx)
add_executable(test_softmax test_softmax.cxx)

add_executable(test_linear test_linear.cxx)
add_executable(test_convolution test_convolution.cxx)
add_executable(test_max_3d_pooling test_max_3d_pooling.cxx)
add_executable(test_max_2d_pooling test_max_2d_pooling.cxx)
add_executable(test_sparse_linear test_sparse_linear.cxx)
add_executable(test_sparse_convolution test_sparse_convolution.cxx)

add_executable(test_neural_network test_neural_network.cxx)