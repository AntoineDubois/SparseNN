import sys
sys.path.append('../build')
import SparseNN

array = SparseNN.Array(2, 3)
def f():
	return 2

array.fill(f)
print(array)

array.fill(3)
print(array)

array2 = SparseNN.Array(2, 4)
array2.fill(4)

array = array2

print("array:", array)
print("array2:", array2)
print("before:", array2[0]) # to pass a tuple to __setitem__ and __getitem__, create python class overload
array2[0] = 2.0
print("after:", array2[0])
print("array2:", array2)


network = SparseNN.NeuralNetwork()
conv_layer = SparseNN.Convolution(1, 1, 3, 1, 0, 1.0, 0.0)
linear_layer = SparseNN.Linear(2, 2, 1.0, 0.0)
flatten_layer = SparseNN.Flatten()
print(flatten_layer(array2))
print(flatten_layer.Forward(array2))
network.append(conv_layer, flatten_layer)
network.append(linear_layer, flatten_layer)

print("before:", network)

sparse_linear_layer = SparseNN.SparseLinear(linear_layer)
network.setLayer(1, sparse_linear_layer)
print("after:", network)

network.clear()


import numpy as np

np_arr = np.array([[1, 2, 3], [1, 2, 3]], dtype=float)
print(np_arr.shape)
print(np_arr.ndim)
pointer, read_only_flag = np_arr.__array_interface__['data']
print(pointer, read_only_flag)
address = np_arr.ctypes.data
print(address)
print(type(address))
#array.from_numpy(np_arr)

def from_numpy(np_array):
	if np_array.ndim == 1:
		lib_array = SparseNN.Array(np_array.shape[0])
	elif np_array.ndim == 2:
		lib_array = SparseNN.Array(np_array.shape[0], np_array.shape[1])
	elif np_array.ndim == 3:
		lib_array = SparseNN.Array(np_array.shape[0], np_array.shape[1], np_array.shape[2])
	else:
		ValueError("The numpy array must have at most 3 dimensions.")

	lib_array.__from_numpy(np_array)
	
	return lib_array

array_copy = from_numpy(np_arr)
np_arr[0] = 2
print(np_arr)
print(array_copy)
print(array_copy.__address())

linear_layer = SparseNN.Linear(6, 1, 1.0, 0.0)
flatten_layer = SparseNN.Flatten()
softmax_layer = SparseNN.Softmax()
array_flat = flatten_layer.Forward(array_copy)
out = linear_layer.Forward(array_flat)
print(out)

print(out.nchannels, out.nrows, out.ncols)

nrows = 6
ncols = 8
n_input_channels = 2
n_output_channels = 4
np_array = np.random.rand(n_input_channels, nrows, ncols)
kernel_size = 3
conv_layer = SparseNN.Convolution(n_input_channels, 1, kernel_size, 1, 0, 1.0, 0.0)
flatten_layer = SparseNN.Flatten()
out_nrows = nrows -kernel_size +1
out_ncols = ncols -kernel_size +1
out_size = out_nrows * out_ncols * n_output_channels
print("out_size=", out_size)
linear_layer = SparseNN.Linear(out_size, 3, 1.0, 0.0)
softmax_layer = SparseNN.Softmax()
identity_layer = SparseNN.Identity()
max_layer = SparseNN.Max()

network = SparseNN.NeuralNetwork()
network.append(conv_layer, identity_layer)
#network.append(linear_layer, max_layer)

lib_arr = from_numpy(np_array)
out = network(lib_arr)
print("out", out)
print(out.size)



np_array = np.random.rand(100)
lib_array = from_numpy(np_array)
linear_layer = SparseNN.Linear(100, 3, 1.0, 0.0)
identity_layer = SparseNN.Identity()
network.clear()
network.append(linear_layer, identity_layer)

out = network(lib_array)
print(out)

