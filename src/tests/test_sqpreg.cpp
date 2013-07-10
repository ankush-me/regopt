#include "sqpreg.hpp"
#include <Eigen/Core>
#include <utils/opt_utils.hpp>

using namespace std;
using namespace Eigen;




void test_sqp_fit_only(unsigned n=200) {

	MatrixXd src = MatrixXd::Random(n,3);
	MatrixXd target = MatrixXd::Random(n,3);

	TPSOptConfig::Ptr config (new TPSOptConfig);
	config->src_pts     = src;
	config->target_pts  = target;
	config->weights     = VectorXd::Ones(n);
	config->rot_coeff   = Vector3d(0.001,0.001,0.00025);
	config->scale_coeff = 0.01;
	config->bend_coeff  = 0.02;
	config->rotreg      = true;

	pair<BasicTrustRegionSQPPtr, TPSOptProb::Ptr> setup =  setup_fit_optimization(config);
	BasicTrustRegionSQPPtr solver = setup.first;
	TPSOptProb::Ptr prob = setup.second;
	solver->optimize();

	vector<double> solution = solver->x();
	MatrixXd B;	extract_vals(solution, B, prob->b_vars);
	MatrixXd c; extract_vals(solution, c, prob->c_vars);
	MatrixXd W; extract_vals(solution, W, prob->w_vars);
	MatrixXd A = prob->N_nq*W;
}


void test_sqp_fit_reg(unsigned n=100 ) {
	MatrixXd src = MatrixXd::Random(n,3);
	MatrixXd target = MatrixXd::Random(n,3);

	RegOptConfig::Ptr config (new RegOptConfig);
	config->src_pts     = src;
	config->target_pts  = target;
	config->rot_coeff   = Vector3d(0.001,0.001,0.00025);
	config->scale_coeff = 0.01;
	config->bend_coeff  = 0.02;
	config->correspondence_coeff = 0.0005;
	config->rotreg      = true;

	pair<BasicTrustRegionSQPPtr, RegOptProb::Ptr> setup =  setup_reg_fit_optimization(config);
	BasicTrustRegionSQPPtr solver = setup.first;
	RegOptProb::Ptr prob = setup.second;
	solver->optimize();

	vector<double> solution = solver->x();
	MatrixXd B;	extract_vals(solution, B, prob->b_vars);
	MatrixXd c; extract_vals(solution, c, prob->c_vars);
	MatrixXd W; extract_vals(solution, W, prob->w_vars);
	MatrixXd A = prob->N_nq*W;
	MatrixXd M; extract_vals(solution, M, prob->m_vars);
}




/**  A small example to show how to initialize a registration problem. */
int main(int cnt, char**v) {
	test_sqp_fit_only();
	return 0;
}
