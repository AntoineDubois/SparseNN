import torch
import os
import sys
from inspect import getsource


class SparseConverter:
    def __init__(self, module : torch.nn.Module, get_max : bool = True, num_format: str = "float", verbose : bool = True) -> None:
        """
        module : torch.nn.Module
        get_max : bool = True # If true, softmax, and log_softmax are replaced with max
        num_format: str = "float" # The C type to use: "float", "double", "long double"
        verbose : bool = True
        """
        print(module)
        if verbose:
            print("The PyTorch module to convert")
            print(getsource(module.forward))
        if not num_format in ["float", "double", "long double"]:
            print("Number format not recognised. Available C types: float, double, long double. But, the function received:", num_format) 
            exit(1)
        self.num_format = num_format

        self.get_max = get_max
        
        
        self._analyse(module)

        if verbose:
            self._print_model()
        
        self._convert()
        self._compile()
        
    def _analyse(self, module : torch.nn.Module ) -> None:
        dict_module = dict(module.named_modules())
        known_keys = list(dict_module.keys())
        known_keys = [string for string in known_keys if string]
        known_values = []
        for key in known_keys:
            known_values.append(dict_module[key])

        forward_str = getsource(module.forward)
        forward_str = forward_str.replace(" ", "")
        forward_str = forward_str.splitlines()        
        forward_str = [string for string in forward_str if string]
        forward_str.pop(0)
        forward_str.pop(-1)


        sparse_network_list = []
        layer_list = ["relu", "log_softmax", "softmax", "max_pool2d", "max_pool3d", "flatten"] # log_softmax must be before softmax, becose softmax is found into softmax
        layer_sparse_name = ["ReLU", "LogSoftmax", "Softmax", "Max2dPooling", "Max3dPooling", "Flatten"]

        for i in range(len(forward_str)):
            forward_str[i] = forward_str[i][forward_str[i].find("=")+1:]
            for j in range(len(known_keys)):
                isfoundmodule = -1 != forward_str[i].find(known_keys[j])
                if isfoundmodule:
                    value = known_values[j]
                    name = value.__class__.__name__
                    if name == "Conv2d":  
                        sparse_name = "Convolution"
                        parameters = {}
                        parameters["input_channels"] = value.in_channels
                        parameters["output_channels"] = value.out_channels
                        parameters["kernel_size"] = value.kernel_size[0]
                        parameters["padding"] = value.padding[0]
                        parameters["stride"] = value.stride[0]
                        parameters["kernel"] = value.weight
                        parameters["bias"] = value.bias
                    elif name == "Linear":
                        sparse_name = "Linear"
                        parameters = {}
                        parameters["input_size"] = value.in_features
                        parameters["output_size"] = value.out_features
                        parameters["weights"] = value.weight
                        parameters["bias"] = value.bias
                    elif name == "MaxPool2d":
                        sparse_name = "Max2dPooling"
                        parameters = {}
                        parameters["kernel_size"] = value.kernel_size
                        parameters["padding"] = value.padding
                        parameters["stride"] = value.stride
                    elif name == "MaxPool3d":
                        sparse_name = "Max3dPooling"
                        parameters = {}
                        parameters["kernel_size"] = value.kernel_size
                        parameters["padding"] = value.padding
                        parameters["stride"] = value.stride
                    else:
                        ValueError("Layer not implemented in SparseNet or not recognised.")
                    
                    sparse_network_list.append({"name": sparse_name, "parameters": parameters})
                    break
            
            for j in range(len(layer_list)):
                isfoundfunctional = -1 != forward_str[i].find(layer_list[j])
                if isfoundfunctional:
                    name = layer_list[j]
                    if name == "max_pool2d":
                        print("max_pool2d has parameters that SparseNet cannot access if defined as a functional in Forward method. Define the max_pool2d in __init__ as a module instead.")
                        exit(1)
                    elif name == "max_pool3d":
                        print("max_pool3d has parameters that SparseNet cannot access if defined as a functional in Forward method. Define the max_pool3d in __init__ as a module instead.")
                        exit(1)
                    elif name in ["softmax", "log_softmax"]:
                        print("Distribution layer replaced with Max layer")
                        parameters = {None: ""}
                        sparse_network_list.append({"name": "Max", "parameters": parameters})
                    else:
                        parameters = {None: ""}
                        sparse_network_list.append({"name": layer_sparse_name[j], "parameters": parameters})
                    break
            
            if not (isfoundmodule or isfoundfunctional):
                print("Layer " + forward_str[i] + " is not implemented in SparseNet or not recognised. Consider defining the functional as a module defined in __init__.") 
                exit(1)
        
        self.sparse_network_list = sparse_network_list

    def _print_model(self) -> None:
        for layer in self.sparse_network_list:
            param_not_weights = layer["parameters"]
            keys = param_not_weights.keys()
            if "kernel" in keys:
                del param_not_weights["kernel"]
            if "weights" in keys:
                del param_not_weights["weights"]
            if "bias" in keys:
                del param_not_weights["bias"]
            print(layer["name"], "\t", param_not_weights)

    def _convert(self):
        model = \
        """
#pragma once
#include "./include/network.hxx"
#include <omp.h>
//#include <pybind11/pybind11.h>
//namespace py = pybind11;

template<typename T>
class CustomNetwork
{{

    NeuralNetwork<{T}> dense_net;
    {declare_layers}
public:
    CustomNetwork(){{
        {new_layers}
        {apppend_layers}
     }};
    Array<{T}> operator()(Array<{T}>& input)
    {{
        omp_set_num_threads(1);
        return dense_net.operator()(input);
    }}
    Array<{T}> Forward(Array<{T}>& input)
    {{
        omp_set_num_threads(1);
        return dense_net.Forward(input);
    }}
    ~CustomNetwork(){{
{del_layers}
    }}
}};
        """
        
        list_layers = []
        model_declare_layers = ""
        model_new_layers = ""
        model_del_layers = ""
        for index, layer in enumerate(self.sparse_network_list):
            (name, isActivation), declare_layer, new_layer, del_layer = self._add_layer(layer, index)
            list_layers.append((name, isActivation))
            model_declare_layers += declare_layer
            model_new_layers += new_layer
            model_del_layers += del_layer
        
        model_append_layers = ""
        i = 0
        while i < len(list_layers):
            layer_name, isActivation = list_layers[i]
            if not isActivation: # the layer is not an activation layer
                next_layer_name, next_isActivation = list_layers[i+1]
                if next_isActivation:
                    model_append_layers += "dense_net.append({layer}, {activation});\n".format(layer=layer_name, activation=next_layer_name)
                    i += 2
                else:
                    model_append_layers += "dense_net.append({layer});\n".format(layer=layer_name)
                    i += 1
            
        model = model.format(T = self.num_format, declare_layers = model_declare_layers, \
                                new_layers = model_new_layers, apppend_layers = model_append_layers, del_layers = model_del_layers)
        
        self.compile_path = os.path.dirname(os.path.abspath(__file__))
        
        self.cwd_path = os.getcwd()
        f = open(self.compile_path + "/" + "custom_model_topython.hxx", "w")
        f.write(model)
        f.close()

        
    def _add_layer(self, layer: dict, index: int) -> str:
        layer_init = "{layer_name}<{T}>* {name}"
        layer_new = "= new {layer_name}<{T}>"
        name = "layer_" + str(index)
        layer_init = layer_init.format(layer_name = layer["name"], name = name, T = self.num_format)
        layer_new = layer_new.format(layer_name = layer["name"], T = self.num_format)
        
        if layer["name"] == "Convolution":
            layer_parameters = "({n_input_channels}, {n_output_channels}, {kernel_size}, {stride}, {padding}, 0.0, 0.0)" # erase 0, 0 when add read weights feature
            layer_parameters = layer_parameters.format(n_input_channels = layer["parameters"]["input_channels"], \
                        n_output_channels = layer["parameters"]["output_channels"], \
                        kernel_size = layer["parameters"]["kernel_size"], \
                        stride = layer["parameters"]["stride"], \
                        padding = layer["parameters"]["padding"])
            isActivation = False
        elif layer["name"] == "Max2dPooling":
            layer_parameters = "({kernel_size}, {stride}, {padding})"
            layer_parameters = layer_parameters.format(kernel_size = layer["parameters"]["kernel_size"], \
                        stride = layer["parameters"]["stride"], \
                        padding = layer["parameters"]["padding"])
            isActivation = False
        elif layer["name"] == "Max3dPooling":
            layer_parameters = "({kernel_size}, {stride}, {padding})"
            layer_parameters = layer_parameters.format(kernel_size = layer["parameters"]["kernel_size"], \
                        stride = layer["parameters"]["stride"], \
                        padding = layer["parameters"]["padding"])
            isActivation = False
        elif layer["name"] == "Linear":
            layer_parameters = "({input_size}, {output_size})"
            layer_parameters = layer_parameters.format(input_size = layer["parameters"]["input_size"], \
                        output_size = layer["parameters"]["output_size"])
            isActivation = False
        else: # the layer has no parameters
            layer_parameters = "()"
            isActivation = True
        layer_new += layer_parameters

        declare_layer = "\n" + layer_init + ";\n"
        new_layer = "\n" + name + layer_new + " ;"
        del_layer = "delete " + name + ";\n"

        return (name, isActivation), declare_layer, new_layer, del_layer

    def _compile(self) -> None:
        commands = \
        """
        pwd;
        cd;
        pwd;
        cd {path};
        pwd;
        mkdir python_build;
        cd python_build;
        cmake ..;
        make DBUILD_PYBIND=true;
        """
        os.system(commands.format(path = self.compile_path.replace(" ", "\ ")))
        

        
    
    