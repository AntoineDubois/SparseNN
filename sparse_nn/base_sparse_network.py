import sys
import os
compile_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(compile_path + "/python_build")
import SparseNN

class BaseSparseNetowrk:
    def __init__(self, num_format: str) -> None:
        if num_format == "float":
            self.network = SparseNN.NeuralNetworkFloat()
        elif num_format == "double":
            self.network = SparseNN.NeuralNetwordDouble()
        elif num_format == "long double":
            self.network = SparseNN.NeuralNetworkLongDouble()
        else:
            print("type not recognised")
            exit(1)
        
    def __call__(self, input_array):
        return self.network.__call__(input_array)

    def Forward(self, input_array):
        return self.network.Forward(input_array)