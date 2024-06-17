from sparse_converter import SparseConverter
from base_sparse_network import BaseSparseNetowrk
import torch

class SparseNetwork(SparseConverter, BaseSparseNetowrk):
    def __init__(self, module: torch.nn.Module, get_max: bool = True, num_format: str = "float", verbose: bool = True) -> None:
        SparseConverter.__init__(self, module, get_max, num_format, verbose)
        BaseSparseNetowrk.__init__(self, num_format)

    def __call__(self, input_array):
        return BaseSparseNetowrk.__call__(self, input_array)

    def Forward(self, input_array):
        return BaseSparseNetowrk.Forward(self, input_array)