#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "./include/algebra.hxx"
#include <sstream>
#include <functional>
#include "custom_model_topython.hxx"

namespace py = pybind11;


PYBIND11_MODULE(SparseNN, handle)
{
    handle.doc() = "The package fastens the inference of pytorch networks";

    // array
    py::class_<Array<float>>(handle, "ArrayFloat")
    .def(py::init<unsigned, unsigned, unsigned>())
    .def(py::init<unsigned, unsigned>())
    .def(py::init<unsigned>())
    .def_readonly("nchannels", &Array<float>::nchannels)
    .def_readonly("nrows", &Array<float>::nrows)
    .def_readonly("ncols", &Array<float>::ncols)
    .def_readonly("size", &Array<float>::size)
    .def("__repr__", [](const Array<float>& array){
        std::stringstream stream;
        stream << array;
        return stream.str();
    })
    .def("fill",
        [](Array<float>& self, const py::function f)
        {
            self.fill(py::cast<const std::function<float()> >(f));
        }
    )
    .def("fill", 
        [](Array<float>& self, float v)
        {
            self.fill(v);
        }
    )
    .def("__from_numpy", [](Array<float>& self, py::array_t<float>& np_array){
        self.values_ptr = (float*)np_array.data();
    }) // not safe but I prefer to keep it for now
    .def("__copy", [](Array<float>& self, py::array_t<float>& np_array){
        for(unsigned e=0; e<self.size; ++e)
        {
            self.values_ptr[e] = *(np_array.data() +e);
        }
    })
    .def("assign", py::overload_cast<Array<float>&>(&Array<float>::operator=) )
    .def("__setitem__", [](Array<float>& self, unsigned index, float value ){
        self.operator()(index) = value;
    })
    .def("__getitem__", [](Array<float>& self, unsigned index){
        return self.operator()(index);
    })
    .def("__address", [](Array<float>& self){
        std::stringstream ss;
        ss << self.values_ptr;
        return ss.str();
    })
    ;
    handle.attr("Array") = handle.attr("ArrayFloat");


    py::class_<Array<double>>(handle, "ArrayDouble")
    .def(py::init<unsigned, unsigned, unsigned>())
    .def(py::init<unsigned, unsigned>())
    .def(py::init<unsigned>())
    .def_readonly("nchannels", &Array<double>::nchannels)
    .def_readonly("nrows", &Array<double>::nrows)
    .def_readonly("ncols", &Array<double>::ncols)
    .def_readonly("size", &Array<double>::size)
    .def("__repr__", [](const Array<double>& array){
        std::stringstream stream;
        stream << array;
        return stream.str();
    })
    .def("fill",
        [](Array<double>& self, const py::function f)
        {
            self.fill(py::cast<const std::function<double()> >(f));
        }
    )
    .def("fill", 
        [](Array<double>& self, double v)
        {
            self.fill(v);
        }
    )
    .def("__from_numpy", [](Array<double>& self, py::array_t<double>& np_array){
        self.values_ptr = (double*)np_array.data();
    }) // not safe but I prefer to keep it for now
    .def("__copy", [](Array<double>& self, py::array_t<double>& np_array){
        for(unsigned e=0; e<self.size; ++e)
        {
            self.values_ptr[e] = *(np_array.data() +e);
        }
    })
    .def("assign", py::overload_cast<Array<double>&>(&Array<double>::operator=) )
    .def("__setitem__", [](Array<double>& self, unsigned index, double value ){
        self.operator()(index) = value;
    })
    .def("__getitem__", [](Array<double>& self, unsigned index){
        return self.operator()(index);
    })
    .def("__address", [](Array<double>& self){
        std::stringstream ss;
        ss << self.values_ptr;
        return ss.str();
    })
    ;

    py::class_<Array<long double>>(handle, "ArrayLongDouble")
    .def(py::init<unsigned, unsigned, unsigned>())
    .def(py::init<unsigned, unsigned>())
    .def(py::init<unsigned>())
    .def_readonly("nchannels", &Array<long double>::nchannels)
    .def_readonly("nrows", &Array<long double>::nrows)
    .def_readonly("ncols", &Array<long double>::ncols)
    .def_readonly("size", &Array<long double>::size)
    .def("__repr__", [](const Array<long double>& array){
        std::stringstream stream;
        stream << array;
        return stream.str();
    })
    .def("fill",
        [](Array<long double>& self, const py::function f)
        {
            self.fill(py::cast<const std::function<long double()> >(f));
        }
    )
    .def("fill", 
        [](Array<long double>& self, long double v)
        {
            self.fill(v);
        }
    )
    .def("__from_numpy", [](Array<long double>& self, py::array_t<long double>& np_array){
        self.values_ptr = (long double*)np_array.data();
    }) // not safe but I prefer to keep it for now
    .def("__copy", [](Array<long double>& self, py::array_t<long double>& np_array){
        for(unsigned e=0; e<self.size; ++e)
        {
            self.values_ptr[e] = *(np_array.data() +e);
        }
    })
    .def("assign", py::overload_cast<Array<long double>&>(&Array<long double>::operator=) )
    .def("__setitem__", [](Array<long double>& self, unsigned index, long double value ){
        self.operator()(index) = value;
    })
    .def("__getitem__", [](Array<long double>& self, unsigned index){
        return self.operator()(index);
    })
    .def("__address", [](Array<long double>& self){
        std::stringstream ss;
        ss << self.values_ptr;
        return ss.str();
    })
    ;
    // linear layer
    py::class_<Linear<float>>(handle, "LinearFloat")
    .def(py::init<unsigned, unsigned, float, float>())
    .def("__call__", &Linear<float>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &Linear<float>::Forward, py::call_guard<py::gil_scoped_release>())
    ;
    handle.attr("Linear") = handle.attr("LinearFloat");

    py::class_<Linear<double>>(handle, "LinearDouble")
    .def(py::init<unsigned, unsigned, double, double>())
    .def("__call__", &Linear<double>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &Linear<double>::Forward, py::call_guard<py::gil_scoped_release>())
    ;

    py::class_<Linear<long double>>(handle, "LinearLongDouble")
    .def(py::init<unsigned, unsigned, long double, long double>())
    .def("__call__", &Linear<long double>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &Linear<long double>::Forward, py::call_guard<py::gil_scoped_release>())
    ;

    // convolution layer
    py::class_<Convolution<float>>(handle, "ConvolutionFloat")
    .def(py::init<unsigned, unsigned, unsigned, unsigned, unsigned, float, float>())
    .def("__call__", &Convolution<float>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &Convolution<float>::Forward, py::call_guard<py::gil_scoped_release>())
    ;
    handle.attr("Convolution") = handle.attr("ConvolutionFloat");

    py::class_<Convolution<double>>(handle, "ConvolutionDouble")
    .def(py::init<unsigned, unsigned, unsigned, unsigned, unsigned, double, double>())
    .def("__call__", &Convolution<double>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &Convolution<double>::Forward, py::call_guard<py::gil_scoped_release>())
    ;

    py::class_<Convolution<long double>>(handle, "ConvolutionLongDouble")
    .def(py::init<unsigned, unsigned, unsigned, unsigned, unsigned, long double, long double>())
    .def("__call__", &Convolution<long double>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &Convolution<long double>::Forward, py::call_guard<py::gil_scoped_release>())
    ;

    // 2D max pooling layer
    py::class_<Max2dPooling<float>>(handle, "Max2dPoolingFloat")
    .def(py::init<unsigned, unsigned, unsigned>())
    .def("__call__", &Max2dPooling<float>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &Max2dPooling<float>::Forward, py::call_guard<py::gil_scoped_release>())
    ;
    handle.attr("Max2dPooling") = handle.attr("Max2dPoolingFloat");

    py::class_<Max2dPooling<double>>(handle, "Max2dPoolingDouble")
    .def(py::init<unsigned, unsigned, unsigned>())
    .def("__call__", &Max2dPooling<double>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &Max2dPooling<double>::Forward, py::call_guard<py::gil_scoped_release>())
    ;

    py::class_<Max2dPooling<long double>>(handle, "Max2dPoolingLongDouble")
    .def(py::init<unsigned, unsigned, unsigned>())
    .def("__call__", &Max2dPooling<long double>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &Max2dPooling<long double>::Forward, py::call_guard<py::gil_scoped_release>())
    ;

    // 3D max pooling layer
    py::class_<Max3dPooling<float>>(handle, "Max3dPoolingFloat")
    .def(py::init<unsigned, unsigned, unsigned>())
    .def("__call__", &Max3dPooling<float>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &Max3dPooling<float>::Forward, py::call_guard<py::gil_scoped_release>())
    ;
    handle.attr("Max3dPooling") = handle.attr("Max3dPoolingFloat");

    py::class_<Max3dPooling<double>>(handle, "Max3dPoolingDouble")
    .def(py::init<unsigned, unsigned, unsigned>())
    .def("__call__", &Max3dPooling<double>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &Max3dPooling<double>::Forward, py::call_guard<py::gil_scoped_release>())
    ;

    py::class_<Max3dPooling<long double>>(handle, "Max3dPoolingLongDouble")
    .def(py::init<unsigned, unsigned, unsigned>())
    .def("__call__", &Max3dPooling<long double>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &Max3dPooling<long double>::Forward, py::call_guard<py::gil_scoped_release>())
    ;

    // Flatten activation layer
    py::class_<Flatten<float>>(handle, "FlattenFloat")
    .def(py::init<>())
    .def("__call__", &Flatten<float>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &Flatten<float>::Forward, py::call_guard<py::gil_scoped_release>())
    ;
    handle.attr("Flatten") = handle.attr("FlattenFloat");

    py::class_<Flatten<double>>(handle, "FlattenDouble")
    .def(py::init<>())
    .def("__call__", &Flatten<double>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &Flatten<double>::Forward, py::call_guard<py::gil_scoped_release>())
    ;

    py::class_<Flatten<long double>>(handle, "FlattenLongDouble")
    .def(py::init<>())
    .def("__call__", &Flatten<long double>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &Flatten<long double>::Forward, py::call_guard<py::gil_scoped_release>())
    ;

    // Identity activation layer
    py::class_<Identity<float>>(handle, "IdentityFloat")
    .def(py::init<>())
    .def("__call__", &Identity<float>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &Identity<float>::Forward, py::call_guard<py::gil_scoped_release>())
    ;
    handle.attr("Identity") = handle.attr("IdentityFloat");

    py::class_<Identity<double>>(handle, "IdentityDouble")
    .def(py::init<>())
    .def("__call__", &Identity<double>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &Identity<double>::Forward, py::call_guard<py::gil_scoped_release>())
    ;

    py::class_<Identity<long double>>(handle, "IdentityLongDouble")
    .def(py::init<>())
    .def("__call__", &Identity<long double>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &Identity<long double>::Forward, py::call_guard<py::gil_scoped_release>())
    ;

    // Log-Softmax activation layer
    py::class_<LogSoftmax<float>>(handle, "LogSoftmaxFloat")
    .def(py::init<>())
    .def("__call__", &LogSoftmax<float>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &LogSoftmax<float>::Forward, py::call_guard<py::gil_scoped_release>())
    ;
    handle.attr("LogSoftmax") = handle.attr("LogSoftmaxFloat");

    py::class_<LogSoftmax<double>>(handle, "LogSoftmaxDouble")
    .def(py::init<>())
    .def("__call__", &LogSoftmax<double>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &LogSoftmax<double>::Forward, py::call_guard<py::gil_scoped_release>())
    ;

    py::class_<LogSoftmax<long double>>(handle, "LogSoftmaxLongDouble")
    .def(py::init<>())
    .def("__call__", &LogSoftmax<long double>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &LogSoftmax<long double>::Forward, py::call_guard<py::gil_scoped_release>())
    ;

    // Softmax activation layer
    py::class_<Softmax<float>>(handle, "SoftmaxFloat")
    .def(py::init<>())
    .def("__call__", &Softmax<float>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &Softmax<float>::Forward, py::call_guard<py::gil_scoped_release>())
    ;
    handle.attr("Softmax") = handle.attr("SoftmaxFloat");

    py::class_<Softmax<double>>(handle, "SoftmaxDouble")
    .def(py::init<>())
    .def("__call__", &Softmax<double>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &Softmax<double>::Forward, py::call_guard<py::gil_scoped_release>())
    ;

    py::class_<Softmax<long double>>(handle, "SoftmaxLongDouble")
    .def(py::init<>())
    .def("__call__", &Softmax<long double>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &Softmax<long double>::Forward, py::call_guard<py::gil_scoped_release>())
    ;

    // max activation layer
    py::class_<Max<float>>(handle, "MaxFloat")
    .def(py::init<>())
    .def("__call__", &Max<float>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &Max<float>::Forward, py::call_guard<py::gil_scoped_release>())
    ;
    handle.attr("Max") = handle.attr("MaxFloat");

    py::class_<Max<double>>(handle, "MaxDouble")
    .def(py::init<>())
    .def("__call__", &Max<double>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &Max<double>::Forward, py::call_guard<py::gil_scoped_release>())
    ;

    py::class_<Max<long double>>(handle, "MaxLongDouble")
    .def(py::init<>())
    .def("__call__", &Max<long double>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &Max<long double>::Forward, py::call_guard<py::gil_scoped_release>())
    ;

    // ReLU activation layer
    py::class_<ReLU<float>>(handle, "ReLUFloat")
    .def(py::init<>())
    .def("__call__", &ReLU<float>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &ReLU<float>::Forward, py::call_guard<py::gil_scoped_release>())
    ;
    handle.attr("ReLU") = handle.attr("ReLUFloat");

    py::class_<ReLU<double>>(handle, "ReLUDouble")
    .def(py::init<>())
    .def("__call__", &ReLU<double>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &ReLU<double>::Forward, py::call_guard<py::gil_scoped_release>())
    ;

    py::class_<ReLU<long double>>(handle, "ReLULongDouble")
    .def(py::init<>())
    .def("__call__", &ReLU<long double>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &ReLU<long double>::Forward, py::call_guard<py::gil_scoped_release>())
    ;

    

    // Neural network
    py::class_<NeuralNetwork<float>>(handle, "NeuralNetworkFloat")
    .def(py::init<>())
    .def("__call__", &NeuralNetwork<float>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &NeuralNetwork<float>::Forward, py::call_guard<py::gil_scoped_release>())
    .def("append", &NeuralNetwork<float>::append)
    ;
    handle.attr("NeuralNetwork") = handle.attr("NeuralNetworkFloat");

    py::class_<NeuralNetwork<double>>(handle, "NeuralNetworkDouble")
    .def(py::init<>())
    .def("__call__", &NeuralNetwork<double>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &NeuralNetwork<double>::Forward, py::call_guard<py::gil_scoped_release>())
    .def("append", &NeuralNetwork<double>::append)
    ;

    py::class_<NeuralNetwork<long double>>(handle, "NeuralNetworkLongDouble")
    .def(py::init<>())
    .def("__call__", &NeuralNetwork<long double>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &NeuralNetwork<long double>::Forward, py::call_guard<py::gil_scoped_release>())
    .def("append", &NeuralNetwork<long double>::append)
    ;    

    // custom compiled neural network
    py::class_<CustomNetwork<float>>(handle, "CustomNeuralNetworkFloat")
    .def(py::init<>())
    .def("__call__", &CustomNetwork<float>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &CustomNetwork<float>::Forward, py::call_guard<py::gil_scoped_release>())
    ;
    handle.attr("CustomNeuralNetwork") = handle.attr("CustomNeuralNetworkFloat");

    py::class_<CustomNetwork<double>>(handle, "CustomNeuralNetworkDouble")
    .def(py::init<>())
    .def("__call__", &CustomNetwork<double>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &CustomNetwork<double>::Forward, py::call_guard<py::gil_scoped_release>())
    ;

    py::class_<CustomNetwork<long double>>(handle, "CustomNeuralNetworkLongDouble")
    .def(py::init<>())
    .def("__call__", &CustomNetwork<long double>::operator(), py::call_guard<py::gil_scoped_release>())
    .def("Forward", &CustomNetwork<long double>::Forward, py::call_guard<py::gil_scoped_release>())
    ;
}