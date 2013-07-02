#pragma once
#include <boost/python.hpp>
#include <vector>
namespace py = boost::python;

py::object np_mod; 

py::list toPyList(const std::vector<int>& x) {
	py::list out;
	for (int i=0; i < x.size(); ++i) out.append(x[i]);
	return out;
}

template<typename T>
struct type_traits {
	static const char* npname;
};
template<> const char* type_traits<float>::npname = "float32";
template<> const char* type_traits<int>::npname = "int32";
template<> const char* type_traits<double>::npname = "float64";
template<> const char* type_traits<unsigned char>::npname = "uint8";

template <typename T>
T* getPointer(const py::object& arr) {
	long int i = py::extract<long int>(arr.attr("ctypes").attr("data"));
	T* p = (T*)i;
	return p;
}

template<typename T>
py::object toNdarray1(const T* data, size_t dim0) {
	py::object out = np_mod.attr("empty")(py::make_tuple(dim0), type_traits<T>::npname);
	T* p = getPointer<T>(out);
	memcpy(p, data, dim0*sizeof(T));
	return out;
}

template<typename T>
py::object toNdarray2(const T* data, size_t dim0, size_t dim1) {
	py::object out = np_mod.attr("empty")(py::make_tuple(dim0, dim1), type_traits<T>::npname);
	T* pout = getPointer<T>(out);
	memcpy(pout, data, dim0*dim1*sizeof(T));
	return out;
}
template<typename T>
py::object toNdarray3(const T* data, size_t dim0, size_t dim1, size_t dim2) {
	py::object out = np_mod.attr("empty")(py::make_tuple(dim0, dim1, dim2), type_traits<T>::npname);
	T* pout = getPointer<T>(out);
	memcpy(pout, data, dim0*dim1*dim2*sizeof(T));
	return out;
}

/** Converts a two-dimensional numpy matrix to eigen MatrixXd. */
Eigen::MatrixXd fromNumpy(py::object mat) {
	assert(("Input numpy matrix is not two dimensional.", py::extract<int>(mat.attr("ndim"))==2));
	mat = np_mod.attr("array")(mat, "float64");
	int rows = py::extract<int>(mat.attr("shape")[0]);
	int cols = py::extract<int>(mat.attr("shape")[1]);

	double* mat_data = getPointer<double>(mat);
	return Eigen::Map<Eigen::MatrixXd>(mat_data, rows, cols);
}

/** Converts a MatrixXd matrix to numpy matrix. */
py::object toNumpy(Eigen:: MatrixXd mat) {
	py::object out     = np_mod.attr("empty")(py::make_tuple(mat.rows(), mat.cols()), "float64");
	double * out_data = getPointer<double>(out);
	Eigen::MatrixXd::Map(out_data, mat.rows(), mat.cols()) = mat;
	return out;
}
