#pragma once

/** Lets you call the python shape-context feature code. */
#include <Eigen/Core>
#include <boost/python.hpp>
#include <limits>

namespace py = boost::python;

const double eps = std::numeric_limits<double>::min();

struct PyGlobals {
	static py::object main_module;
	static py::object main_namespace;
	static py::object builtin_module;
	static py::object shape_context_module;
	static py::object numpy_module;
	static py::object None;
};
#define np PyGlobals::numpy_module;

/** Initializes the python modules. */
void setup_python();


/** Wrapper around python's shape-context implementation.
 *  @params:
 *     - pts : nx3 matrix of points
 *     - r_inner, r_outer: min/max (radius/median_radius) ratio for binning
 *                         bins are : [0, r_inner, ...., r_outer]
 *	   - nbins_{r,theta,phi}  : number of bins for the polar coordinates. */
Eigen::MatrixXd shape_context(Eigen::MatrixXd pts,
		double r_inner=1./8., double r_outer=2.,
		int nbins_r = 5, int nbins_theta=12, int nbins_phi=6);


/** Calculates the pairwise shape-distance b/w two clouds' shape-context histograms.*/
Eigen::MatrixXd shape_distance(const Eigen::MatrixXd &sc1_nb, const Eigen::MatrixXd &sc2_mb);
