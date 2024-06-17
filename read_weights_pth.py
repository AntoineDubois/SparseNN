import torch
from torchvision import datasets, transforms
from time import time

model_weights = torch.load("./weights/mnist_cnn.pth")

#print(model_weights) # in only a dictonarry

print(model_weights.keys())

model_sript = torch.jit.load("./weights/mnist_cnn.pt")
print(type(model_sript))
#print(model_sript.graph)
transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)

test_batch_size = 1000
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=test_batch_size)

rd_element = int( torch.randint(low=0, high=len(dataset2), size=(1,)) )
observation = dataset2[rd_element]
x = observation[0]
y = observation[1]
x = x[None]

y_hat = model_sript(x)
start = time()
y_hat = model_sript(x)
y_hat = y_hat.argmax()
end = time()
print("TORCH prediction:", y_hat, ", true value:", y, ", duration:", end -start)

model_compiled = torch.compile(model_sript)
start = time()
y_hat = model_compiled(x)
y_hat = y_hat.argmax()
end = time()
print("COMPILED prediction:", y_hat, ", true value:", y, ", duration:", end -start)


import sys
sys.path.append('./sparse_nn/build')
import SparseNN

conv_layer1 = SparseNN.Convolution(1, 32, 3, 1, 0, 1.0, 0.0)
conv_layer2 = SparseNN.Convolution(32, 64, 3, 1, 0, 1.0, 0.0)
linear_layer1 = SparseNN.Linear(9216, 128, 1.0, 0.0)
linear_layer2 = SparseNN.Linear(128, 10, 1.0, 0.0)

flatten_layer = SparseNN.Flatten()
relu_layer = SparseNN.ReLU()
max_pool_layer = SparseNN.Max2dPooling(2, 2, 0)
max_layer = SparseNN.Max()

network = SparseNN.NeuralNetwork()
network.append(conv_layer1, relu_layer)
network.append(conv_layer2, relu_layer)
network.append(max_pool_layer, flatten_layer)
network.append(linear_layer1, relu_layer)
network.append(linear_layer2, max_layer)

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

x = observation[0]
y = observation[1]

x_np = x.numpy()
x_lib = from_numpy(x_np)
start = time()
y_hat_lib = network(x_lib)
end = time()
print("SPARSE prediction:", y_hat_lib, ", true value:", y, ", duration:", end -start)

start = time()
y_hat_lib = network.Forward(x_lib)
end = time()
print("SPARSE prediction:", y_hat_lib, ", true value:", y, ", duration:", end -start)

del conv_layer1
del conv_layer2
del linear_layer1
del linear_layer2

del flatten_layer
del relu_layer
del max_pool_layer
del max_layer

del y_hat_lib
del y
del x_lib
del network

print("here")
exit(0)
