#include <Eigen/Core>
#include <boost/python.hpp>
#include "numpy_utils.hpp"

#include "sqpreg.hpp"
#include "problem_description.hpp"
#include "utils/opt_utils.hpp"
#include <utility>

namespace py = boost::python;
using namespace Eigen;
using namespace std;

/** Lets python call the sqp based registration code.
 *  Returns the non-linear weights, affine and translation part.
 *
 *  Parameters: source point-cloud, target point-cloud, optimization costs.*/
py::object fit_tps_sqp(py::object src_cloud, py::object target_cloud,
		py::object rot_coeff, double scale_coeff, double bend_coeff,
		double corr_coeff, bool rotreg=true) {

	RegOptConfig::Ptr config (new RegOptConfig);
	config->src_pts     = fromNumpy(src_cloud);
	config->target_pts  = fromNumpy(target_cloud);
	config->rot_coeff   = fromNumpyVector(rot_coeff);
	config->scale_coeff = scale_coeff;
	config->bend_coeff  = bend_coeff;
	config->correspondence_coeff = corr_coeff;
	config->rotreg      = rotreg;

	cout << "============================================\n"
			"             C++ Info                       \n"
			"============================================\n";
	cout << "source pts: "<< endl << config->src_pts <<endl;
	cout << "target pts: "<< endl << config->target_pts <<endl;
	cout << "rot coeffs: "<< config->rot_coeff.transpose() <<endl;
	cout << "scale coeff: "<< config->scale_coeff<<endl;
	cout << "bend coeff: " << config->bend_coeff <<endl;
	cout << "corres coeff: "<< config->correspondence_coeff <<endl;
	cout << "rot reg: "<<  config->rotreg <<endl;
	cout << "===================================================\n";


	pair<BasicTrustRegionSQPPtr, RegOptProb::Ptr> opt_prob =  setup_optimization(config);
	BasicTrustRegionSQPPtr solver = opt_prob.first;
	RegOptProb::Ptr prob          = opt_prob.second;

	solver->optimize();

	vector<double> solution = solver->x();
	MatrixXd B;	extract_vals(solution, B, prob->b_vars);
	MatrixXd c; extract_vals(solution, c, prob->c_vars);
	MatrixXd W; extract_vals(solution, W, prob->w_vars);
	MatrixXd A = prob->N_nq*W;
	MatrixXd M; extract_vals(solution, M, prob->m_vars);

	cout << "============================================\n"
			"             C++ Info                       \n"
			"============================================\n";
	cout << "tps weights : " << A<<endl;
	cout << "linear : " << B<<endl;
	cout << "translation : " << c<<endl;
	cout << "correspondences : " << M <<endl;
	cout << "===================================================\n";

	cout << "call count: "<< prob->call_count<<endl;
	cout << "num vars: "<< prob->getVars().size()<<endl;

	py::list out;
	out.append(toNumpy(A));
	out.append(toNumpy(B));
	out.append(toNumpy(c));

	return out;
}

BOOST_PYTHON_MODULE(sqpregpy) {
	np_mod = py::import("numpy");

	py::def("fit_tps_sqp", &fit_tps_sqp, (py::arg("src"), py::arg("target"), py::arg("rot_coeff"),
			py::arg("scale_coeff"), py::arg("bend_coeff"), py::arg("correspondence_coeff"), py::arg("rotreg")));
}
