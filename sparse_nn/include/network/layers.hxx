#pragma once

// activation layers
#include "./activation_layers/base_activation_layer.hxx"
#include "./activation_layers/flatten.hxx"
#include "./activation_layers/identity.hxx"
#include "./activation_layers/logsoftmax.hxx"
#include "./activation_layers/max.hxx"
#include "./activation_layers/relu.hxx"
#include "./activation_layers/softmax.hxx"

// layers
#include "./layers/base_layer.hxx"
#include "./layers/base_convolution.hxx"
#include "./layers/convolution.hxx"
#include "./layers/linear.hxx"
#include "./layers/max_2d_pooling.hxx"
#include "./layers/max_3d_pooling.hxx"
#include "./layers/sparse_convolution.hxx"
#include "./layers/sparse_linear.hxx"
