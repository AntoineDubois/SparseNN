import sys
sys.path.append('./sparse_nn/build')
import SparseNN

conv_layer1 = SparseNN.Convolution(1, 32, 3, 1, 0, 1.0, 0.0)
conv_layer2 = SparseNN.Convolution(32, 64, 3, 1, 0, 1.0, 0.0)
linear_layer1 = SparseNN.Linear(9216, 128, 1.0, 0.0)
linear_layer2 = SparseNN.Linear(128, 10, 1.0, 0.0)

flatten_layer = SparseNN.Flatten()
relu_layer = SparseNN.ReLU()
max_pool_layer = SparseNN.Max2dPooling(2, 1, 0)
log_softmax_layer = SparseNN.LogSoftmax()

network = SparseNN.NeuralNetwork()
network.append(conv_layer1, relu_layer)
#network.append(conv_layer2, relu_layer)
#network.append(max_pool_layer, flatten_layer)
#network.append(linear_layer1, relu_layer)
#network.append(linear_layer2, log_softmax_layer)

def from_numpy(np_array):
	if np_array.ndim == 1:
		lib_array = SparseNN.Array(np_array.shape[0])
	elif np_array.ndim == 2:
		lib_array = SparseNN.Array(np_array.shape[0], np_array.shape[1])
	elif np_array.ndim == 3:
		lib_array = SparseNN.Array(np_array.shape[0], np_array.shape[1], np_array.shape[2])
	else:
		ValueError("The numpy array must have at most 3 dimensions.")

	#lib_array.__from_numpy(np_array)
	lib_array.__copy(np_array)
	return lib_array