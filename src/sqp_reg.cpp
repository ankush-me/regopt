#include "problem_description.hpp"
#include "utils/eigen_utils.hpp"
#include <sco/optimizers.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <algorithm>
#include <utility>

using namespace Eigen;
using namespace std;
using namespace sco;

typedef boost::shared_ptr<BasicTrustRegionSQP> BasicTrustRegionSQPPtr;

// put the values in the proper place in the big solution vector X.
void set_vals(vector<double> &x, const MatrixXd & data, const VarArray &index_mat) {
	assert(("Indexing array and data array should have same shape. ",
			data.cols()==index_mat.cols() && data.rows()==index_mat.rows()));

	for(unsigned i =0; i < data.rows(); i +=1) {
		for(unsigned j=0; j < data.cols(); j+=1) {
			x[index_mat(i,j).var_rep->index] = data(i,j);
		}
	}
}


/** - Creates a registration problem.
 *    - adds it to the basic trust region solver
 *      - initializes the solution vector appropriately.
 *
 *   Returns a pointer to the sqp optimizer.*/
pair<BasicTrustRegionSQPPtr, RegOptProb::Ptr> setup_optimization(RegOptConfig::Ptr reg_config) {

	RegOptProb::Ptr prob(new RegOptProb(reg_config));
	BasicTrustRegionSQPPtr solver(new BasicTrustRegionSQP(prob));
	solver->trust_box_size_ = 100;


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


int main(int c, char**v) {

	MatrixX3d src(4,3), target(10,3);
	src << 1,0,1, -1,2,2, 0,0,0, 3,3,3;

	RegOptConfig::Ptr config (new RegOptConfig);
	config->src_pts     = src;
	config->target_pts  = target;
	config->rot_coeff   = Vector3d(1,1,1);
	config->scale_coeff = 1;
	config->bend_coeff  = 1;
	config->correspondence_coeff = 1;
	config->rotreg      = true;

	pair<BasicTrustRegionSQPPtr, RegOptProb::Ptr> setup =  setup_optimization(config);
	RegOptProb::Ptr prob = setup.second;

	cout << "m_vars : "<<endl;
	for (int i = 0; i < prob->m_vars.rows(); i+=1) {
		for (int j=0; j < prob->m_vars.cols(); j+=1) {
			cout << " " << prob->m_vars(i,j).var_rep->name;
		}
		cout <<endl;
	}

	cout << "a_vars : "<<endl;
	for (int i = 0; i < prob->a_vars.rows(); i+=1) {
		for (int j=0; j < prob->a_vars.cols(); j+=1) {
			cout << " " << prob->a_vars(i,j).var_rep->name;
		}
		cout <<endl;
	}


	return 0;
}
