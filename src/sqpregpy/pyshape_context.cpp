/** Lets you call the python shape-context feature code. */
#include "pyshape_context.hpp"
#include "numpy_utils.hpp"
#include <assert.h>
#include <utils/colorize.h>
#include <iostream>

using namespace Eigen;
using namespace std;

py::object PyGlobals::shape_context_module;
py::object PyGlobals::numpy_module;
py::object PyGlobals::None;
py::object PyGlobals::main_module;
py::object PyGlobals::main_namespace;
py::object PyGlobals::builtin_module;

/** Initializes the python modules.
 *  This function makes sure that the python setup is called once only. */
void setup_python() {
	static bool initialized = false;
	if (!initialized) {
		cout << colorize("Setting up python modules.", "green", true) << endl;
		try {
			Py_Initialize();

			PyGlobals::main_module    = py::import("__main__");
			PyGlobals::main_namespace = PyGlobals::main_module.attr("__dict__");
			PyGlobals::builtin_module = py::import("__builtin__");

			py::exec("import sys", PyGlobals::main_namespace);
			py::exec("sys.argv = ['call_shape_context_from_cpp']", PyGlobals::main_namespace);

			PyGlobals::shape_context_module = py::import("reg_clouds.shape_context");
			PyGlobals::numpy_module  = py::import("numpy");
			PyGlobals::None = py::api::object();
			np_mod = PyGlobals::numpy_module;
		} catch(...) {
			PyErr_Print();
		}
		initialized = true;
	}
}


/** Wrapper around python's shape-context implementation.
 *  @params:
 *     - pts : nx3 matrix of points
 *     - r_inner, r_outer: min/max (radius/median_radius) ratio for binning
 *                         bins are : [0, r_inner, ...., r_outer]
 *	   - nbins_{r,theta,phi}  : number of bins for the polar coordinates. */
MatrixXd shape_context(MatrixXd pts,
		double r_inner, double r_outer,
		int nbins_r, int nbins_theta, int nbins_phi) {
	setup_python();
	try {
		py::object sc_func = PyGlobals::shape_context_module.attr("shape_context");
		py::object sc_features = sc_func(toNP(pts), PyGlobals::None, r_inner, r_outer,
				nbins_r, nbins_theta, nbins_phi, PyGlobals::None, true, false);
		return fromNP(sc_features);
	} catch (...) {
		PyErr_Print();
	}
}


/** Calculates the pairwise shape-distance b/w two clouds' shape-context histograms.*/
MatrixXd shape_distance(const MatrixXd &sc1_nb, const MatrixXd &sc2_mb) {
	if (sc1_nb.cols() != sc2_mb.cols()) {
		cout << colorize("Shape distance error : Source and target clouds have different bin sizes.", "red", true);
		return MatrixXd();
	}

	const unsigned b = sc1_nb.cols();
	const unsigned n = sc1_nb.rows();
	const unsigned m = sc2_mb.rows();

	// normalize the histograms
	VectorXd rowsum1 = sc1_nb.rowwise().sum().array() + eps;
	MatrixXd sc1_norm_nb(n, b);
	for(unsigned i=0; i < n; i+=1)
		sc1_norm_nb.row(i) = sc1_nb.row(i)/rowsum1[i];

	VectorXd rowsum2 = sc2_mb.rowwise().sum().array() + eps;
	MatrixXd sc2_norm_mb(m,b);
	for(unsigned i=0; i < m; i+=1)
		sc2_norm_mb.row(i) = sc2_mb.row(i)/rowsum2[i];

	MatrixXd d(n,m);
	for(unsigned i=0; i < n; i+=1) {
		d.row(i) = 0.5*((sc2_norm_mb.rowwise() - sc1_norm_nb.row(i)).array().square()
						/ ((sc2_norm_mb.rowwise() + sc1_norm_nb.row(i)).array() + eps));
	}
	return d;
}
