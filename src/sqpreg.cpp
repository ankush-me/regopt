#include "sqpreg.hpp"
#include "utils/eigen_utils.hpp"
#include <algorithm>


using namespace Eigen;
using namespace std;
using namespace sco;



/** - Creates a registration problem.
 *    - adds it to the basic trust region solver
 *      - initializes the solution vector appropriately.
 *
 *   Returns a pointer to the sqp optimizer.*/
pair<BasicTrustRegionSQPPtr, RegOptProb::Ptr> setup_optimization(RegOptConfig::Ptr reg_config) {

	RegOptProb::Ptr prob(new RegOptProb(reg_config));
	BasicTrustRegionSQPPtr solver(new BasicTrustRegionSQP(prob));
	solver->trust_box_size_ = 100;
	solver->max_iter_ = 20;
	solver->min_trust_box_size_ = 1e-6;

	/** Initialize the solution vector:
	 *
	 *   1. The correspondence matrix (M) is initialized to uniform weights,
	 *      such that the rows sum to one.
	 *   2. The affine part (B) is initialized to identity.
	 *   3. The translation (c) is initialized to the
	 *      difference of the median vectors of the
	 *      source and the target point clouds.
	 *   4. The weight matrix (A) is initialized to all ones. */

	vector<double> x(prob->getNumVars(), 0);
	// 1. correspondence matrix
	MatrixXd M = MatrixXd::Ones(prob->m_vars.rows(), prob->m_vars.cols());
	M /= (M.cols()-1.0);
	M.col(M.cols()-1) = VectorXd::Zero(M.rows());
	M.row(M.rows()-1) = VectorXd::Zero(M.cols());
	set_vals(x, M, prob->m_vars);

	// 2. affine part : identity
	MatrixXd B = Matrix3d::Identity();
	set_vals(x, B, prob->b_vars);

	// 3. translation : difference of medians
	VectorXd src_median    = median(prob->src_nd, 0);
	VectorXd target_median = median(prob->target_md, 0);
	MatrixXd c(3,1);
	c = target_median - src_median;
	set_vals(x, c, prob->c_vars);

	// 4. weight matrix
	MatrixXd A = MatrixXd::Ones(prob->a_vars.rows(), prob->a_vars.cols());
	set_vals(x, A, prob->a_vars);

	solver->initialize(x);
	return make_pair(solver, prob);
}