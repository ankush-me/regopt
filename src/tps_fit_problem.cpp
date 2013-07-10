#include "tps_fit_problem.hpp"
#include <boost/bind.hpp>
#include <assert.h>


#include <sco/modeling_utils.hpp>
#include <sco/expr_ops.hpp>
#include <utils/opt_utils.hpp>

#include "rotreg.hpp"
#include "tps_costs.hpp"

using namespace Eigen;
using namespace sco;
using namespace std;

TPSOptProb::TPSOptProb(TPSOptConfig::Ptr config) :
		src_nd(config->src_pts),
		target_nd(config->target_pts),
		w_n(config->weights),
		n_pts(config->src_pts.rows()),
		rot_coeff(config->rot_coeff),
		scale_coeff(config->scale_coeff),
		bend_coeff(config->bend_coeff),
		rotreg(config->rotreg) {

	init();
}

TPSOptProb::TPSOptProb(const MatrixX3d &src_pts, const MatrixX3d &target_pts,
		const VectorXd &weights,
		const Vector3d &rot_coeff_, double scale_coeff_, double bend_coeff_, bool rotreg_) :
		src_nd(src_pts), target_nd(target_pts),
		w_n(weights),
		n_pts(src_pts.rows()),
		rot_coeff(rot_coeff_), scale_coeff(scale_coeff_), bend_coeff(bend_coeff_),
		rotreg(rotreg_) {
	init();
}

/** Does self-initialization. */
void TPSOptProb::init() {
	assert(("Points are expected to be three dimensional", src_nd.cols()==3 && target_nd.cols()==3));
	assert(("Same number of source and target points expected.", src_nd.rows()==target_nd.rows()));

	call_count = 0;

	// compute the tps-kernel matrix
	MatrixXd K_nn(n_pts, n_pts);
	for (int i=0; i < n_pts; i++)
		K_nn.row(i) = (src_nd.rowwise() - src_nd.row(i)).rowwise().norm();

	// Calculate the null-space of [X 1].T and project A on that.
	// by doing this, the vanishing_moment_constraints are not needed.
	MatrixXd X1(src_nd.rows(), src_nd.cols()+1);
	X1 << MatrixXd::Ones(src_nd.rows(),1), src_nd;
	JacobiSVD<MatrixXd> svd(X1, ComputeFullU);
	N_nq = svd.matrixU().block(0,4,src_nd.rows(), src_nd.rows()-4);
	KN_nq = K_nn*N_nq;

	init_vars();
	init_costs();
	init_constraints();
}

/** Initialize the optimization variables.
 *
 *  say N = n_src, M = m_target
 *
 *  The variables are:
 *  ==================
 *    1. c (3x1)          : translation
 *    2. B (3x3)          : affine part
 *    3. W ((N-4)x3)      : tps kernel weights of null of [X 1] */
void TPSOptProb::init_vars() {

	// translation variables
	unsigned int last_size = getVars().size();
	vector<string> c_names(3);
	c_names[0] = "c_x"; c_names[0] = "c_y";  c_names[0] = "c_z";
	createVariables(c_names);
	c_vars = VarArray(3, 1, getVars().data()+last_size);


	// affine variables
	last_size = getVars().size();
	vector<string> b_names;
	for (unsigned i=0; i < 3; i+=1) {
		for(int j=0; j<3; j+=1) {
			b_names.push_back((boost::format("b_%i_%i")%i%j).str());
		}
	}
	createVariables(b_names);
	b_vars = VarArray(3, 3, getVars().data()+last_size);


	// tps kernel weights [weights of Null-space of [X 1]]
	// such that A = N_nq*w_q3
	last_size = getVars().size();
	vector<string> w_names(KN_nq.cols()*3);
	for (unsigned i=0; i < KN_nq.cols(); i+=1) {
		for(int j=0; j < 3; j+=1) {
			w_names[i*3 + j] = (boost::format("w_%i_%i")%i%j).str();
		}
	}
	createVariables(w_names);
	w_vars = VarArray(KN_nq.cols(), 3, getVars().data()+last_size);
}

void  TPSOptProb::init_costs() {
	// add tps-cost
	addCost(CostPtr(new ResidualCost(this)));

	// add the bending cost:
	addCost(CostPtr(new BendingCost(KN_nq, N_nq, bend_coeff, w_vars)));

	if (rotreg) {
		cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ADDING ROT REG <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";
		ScalarOfVectorPtr f_rotreg_ptr = ScalarOfVector::construct(boost::bind( &TPSOptProb::f_rotreg_cost, this, _1 ));
		addCost(CostPtr(new CostFromFunc(f_rotreg_ptr, b_vars.m_data, "f_rotreg_cost", true)));
	}
}

/** Adds the following constraints:
 *  =============================== */
void  TPSOptProb::init_constraints() {}


/** Computes the polar-decomposition cost. Uses rapprentice's fastrapp. */
double  TPSOptProb::f_rotreg_cost(const VectorXd& x) {
	Map<Matrix3d> B_33((double*)x.data(),3,3);
	return RotReg(B_33, rot_coeff, scale_coeff);
}
