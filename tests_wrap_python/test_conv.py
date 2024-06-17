import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# With one input channel and one output channel
print("With 1 input channel and 1 output channel:")
kernel_size = 3
stride = 1
padding = 1
n = 10
m = 6
input = torch.randn(1, 1, n, m)
layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
output = layer(input)

def convolution(input, kernel, kernel_size, padding, stride, bias = 0.0):
    n = input.size(0)
    m = input.size(1)
    a = int( ( n + 2 * padding -kernel_size)/stride) +1
    b = int( ( m + 2 * padding -kernel_size)/stride) +1
    output = torch.zeros((a, b))

    for i in range(0, a):
        for j in range(0, b):
            s = 0.0
            i_prime = i * stride - padding
            j_prime = j * stride - padding
            
            for u in range(max(0, -i_prime), min(kernel_size, n -i_prime)):
                for v in range(max(0, -j_prime), min(kernel_size, m -j_prime)):
                    s += input[i_prime +u, j_prime +v] * kernel[u, v]

            output[i, j] = s + bias
    return output

input_scratch = input.flatten(0, 2)
kernel = layer.weight.flatten(0, 2)
bias = float(layer.bias)

out_scratch = convolution(input=input_scratch, kernel=kernel, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)

print( torch.max( torch.abs(out_scratch -output) ) )


# With two input channels and one output channel
print("\nWith 2 input channels and 1 output channel:")
layer = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
input = torch.randn(1, 2, n, m)
output = layer(input)
input_scratch_ch1 = input[:,0].flatten(0, 1)
input_scratch_ch2 = input[:,1].flatten(0, 1)

kernel_ch1 = layer.weight[:,0,:,:].flatten(0,1)
kernel_ch2 = layer.weight[:,1,:,:].flatten(0,1)
bias = float(layer.bias)


out_scratch_ch1 = convolution(input=input_scratch_ch1, kernel=kernel_ch1, kernel_size=kernel_size, padding=padding, stride=stride, bias=0.0)
out_scratch_ch2 = convolution(input=input_scratch_ch2, kernel=kernel_ch2, kernel_size=kernel_size, padding=padding, stride=stride, bias=0.0)

out_scratch = out_scratch_ch1 + out_scratch_ch2
out_scratch += bias # same bias for all chanels, the bias is added to the sum of the filders
print( torch.max( torch.abs(out_scratch -output) ) )

# With one input channel and two output channels
print("\nWith 1 input channel and 2 output channels:")
layer = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
input = torch.randn(1, 1, n, m)
output = layer(input)
output1 = output[:,0,:,:].flatten(0,1)
output2 = output[:,1,:,:].flatten(0,1)
input_scratch = input.flatten(0, 2)

kernel_out1 = layer.weight[0,:,:,:].flatten(0,1)
kernel_out2 = layer.weight[1,:,:,:].flatten(0,1)
bias_out1 = float(layer.bias[0])
bias_out2 = float(layer.bias[1])

out_scratch_out1 = convolution(input=input_scratch, kernel=kernel_out1, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias_out1)
out_scratch_out2 = convolution(input=input_scratch, kernel=kernel_out2, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias_out2)

print( torch.max( torch.abs(out_scratch_out1 -output1) ) )
print( torch.max( torch.abs(out_scratch_out2 -output2) ) )

# With two input channels and two output channels
print("\nWith 2 input channels and 2 output channels:")

layer = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)

input = torch.randn(1, 2, n, m)
output = layer(input)

print(output.size())

output1 = output[:,0,:,:].flatten(0,1)
output2 = output[:,1,:,:].flatten(0,1)

input_scratch_ch1 = input[:,0].flatten(0, 1)
input_scratch_ch2 = input[:,1].flatten(0, 1)

# [out_channels, in_channels, kernel_i, kernel_j]
kernel_ch1_out1 = layer.weight[0,0,:,:]
kernel_ch1_out2 = layer.weight[1,0,:,:]
kernel_ch2_out1 = layer.weight[0,1,:,:]
kernel_ch2_out2 = layer.weight[1,1,:,:]
bias_out1 = float(layer.bias[0])
bias_out2 = float(layer.bias[1])


out_scratch_ch1_out1 = convolution(input=input_scratch_ch1, kernel=kernel_ch1_out1, kernel_size=kernel_size, padding=padding, stride=stride, bias=0.0)
out_scratch_ch1_out2 = convolution(input=input_scratch_ch1, kernel=kernel_ch1_out2, kernel_size=kernel_size, padding=padding, stride=stride, bias=0.0)
out_scratch_ch2_out1 = convolution(input=input_scratch_ch2, kernel=kernel_ch2_out1, kernel_size=kernel_size, padding=padding, stride=stride, bias=0.0)
out_scratch_ch2_out2 = convolution(input=input_scratch_ch2, kernel=kernel_ch2_out2, kernel_size=kernel_size, padding=padding, stride=stride, bias=0.0)
out_scratch_out1 = out_scratch_ch1_out1 + out_scratch_ch2_out1 + bias_out1
out_scratch_out2 = out_scratch_ch1_out2 + out_scratch_ch2_out2 + bias_out2

print(layer.bias)

print( torch.max( torch.abs(out_scratch_out1 -output1) ) )
print( torch.max( torch.abs(out_scratch_out2 -output2) ) )