import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

kernel_size = 3
stride = 1
padding = 1
n = 10
m = 6

n_output_channels = 1
n_input_channels = 2
input = torch.rand(size=(n_output_channels, n_input_channels, n, m)) -0.5

print(input)


layer = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
output = layer(input)
print(output)

#print(input)
#print(output)

print(input.size())
print(output.size())

layer = nn.MaxPool3d(kernel_size, stride=stride, padding=padding)
output_3d = layer(input)

#print(input)
#print(output)

print(input.size())
print(output_3d.size())

print(torch.max(torch.abs(output -output_3d)))

print(output[0,0,:, :])
print(output[0,1,:, :])
print("Assertionn: Conv2d takes the maximum of block on a channel")

print(output_3d[0,0,:, :])
print(output_3d[0,1,:, :])
print("Assertion: Conv3d takes the maximum of a block over all the channels")
print("The two output channels of output_3d are identical: max absolute difference=", torch.max(torch.abs(output_3d[0,0,:,:] -output_3d[0,1,:,:])))

max_over_the_channels = torch.maximum(output[0,0,:, :], output[0,1,:, :])

print("The assumption is true: max absolution difference=", torch.max(torch.abs(max_over_the_channels -output_3d[0,0,:,:])))