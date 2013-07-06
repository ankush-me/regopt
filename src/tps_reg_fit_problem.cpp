#include "tps_reg_fit_problem.hpp"
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <boost/bind.hpp>

// John's sequential convex optimizer stuff
#include <sco/modeling_utils.hpp>
#include <sco/expr_ops.hpp>

#include "rotreg.hpp"
#include <stdlib.h>



using namespace Eigen;
using namespace std;
using namespace sco;


RegOptProb::RegOptProb(RegOptConfig::Ptr config) :
								src_nd(config->src_pts), target_md(config->target_pts),
								rot_coeff(config->rot_coeff), scale_coeff(config->scale_coeff),
								bend_coeff(config->bend_coeff),
								correspondence_coeff(config->correspondence_coeff),
								n_src(src_nd.rows()), m_target(target_md.rows()),
								rotreg(config->rotreg) {
	init();
}


RegOptProb::RegOptProb(const MatrixX3d &src_pts, const MatrixX3d &target_pts,
		const Vector3d &rot_coeff_, double scale_coeff_, double bend_coeff_,
		double correspondence_coeff_, bool rotreg_)
: src_nd(src_pts), target_md(target_pts),
  rot_coeff(rot_coeff_), scale_coeff(scale_coeff_), bend_coeff(bend_coeff_),
  correspondence_coeff(correspondence_coeff_),
  n_src(src_nd.rows()), m_target(target_md.rows()),
  rotreg(rotreg_) {

	init();
}

/** Does self-initialization. */
void RegOptProb::init() {

	call_count = 0;

	// compute the tps-kernel matrix
	MatrixXd K_nn(n_src, n_src);
	for (int i=0; i < n_src; i++)
		K_nn.row(i) = (src_nd.rowwise() - src_nd.row(i)).rowwise().norm();

	// Calculate the null-space of [X 1].T and project A on that.
	// by doing this, the vanishing_moment_constraints are not needed.
	MatrixXd X1(src_nd.rows(), src_nd.cols()+1);
	X1 << src_nd, MatrixXd::Ones(src_nd.rows(),1);
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
 *    1. M ((M+1)x (N+1)) : correspondence matrix
 *    2. c (3x1)          : translation
 *    3. B (3x3)          : affine part
 *    4. W ((N-4)x3)      : tps kernel weights of null of [X 1]
 */
void RegOptProb::init_vars() {
	// correspondence matrix variables
	unsigned int m_size = (n_src+1)*(m_target+1);
	vector<string> m_names(m_size);
	for (unsigned i=0; i < m_target+1; i+=1) {
		for (unsigned j=0; j < n_src+1; j+=1) {
			m_names[i*(n_src+1) + j] =  (boost::format("m_%i_%i")%i%j).str();
		}
	}
	createVariables(m_names, vector<double>(m_size, 0), vector<double>(m_size, 1));
	m_vars = VarArray((m_target+1), (n_src+1), getVars().data());


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

void  RegOptProb::init_costs() {
	// add tps-cost
	addCost(CostPtr(new WeightedResidualCost(this)));

	// add the bending cost:
	addCost(CostPtr(new BendingCost(KN_nq, N_nq, bend_coeff, w_vars)));

	// add the cost on correspondence matrix:
	addCost(CostPtr(new CorrespondenceCost(correspondence_coeff, m_vars.block(0,0, m_target, n_src))));

	if (rotreg) {
		ScalarOfVectorPtr f_rotreg_ptr = ScalarOfVector::construct(boost::bind( &RegOptProb::f_rotreg_cost, this, _1 ));
		addCost(CostPtr(new CostFromFunc(f_rotreg_ptr, b_vars.m_data, "f_rotreg_cost", true)));
	}
}

/** Adds the following constraints:
 *  ===============================
 *    1. Doubly-stochastic constraint on M (allowing for outliers).
 *    2. (1 X).T * A = 0. : not needed as A is projected onto the null space of [1 X].
 *  M_ij \in [0,1] : this is already set in variable construction. */
void  RegOptProb::init_constraints() {
	doubly_stochastic_constraints();
}

/** Computes the tps-cost, given the current long solution vector x.
 *  Will use numerical differentiation for convex approximation.*/
double  RegOptProb::f_tps_cost(const VectorXd& x) {
	call_count += 1;

	// retrieve current values in the correct matrix shapes.
	MatrixXd M_mn = getMat(x, m_vars.block(0,0, m_target, n_src));
	Vector3d c_3  = Vector3d(getMat(x, c_vars));
	MatrixXd B_33 = getMat(x, b_vars);
	MatrixXd W_q3 = getMat(x, w_vars);

	// error term: sum_ij M_ij * || T_i - estimate_j ||^2
	double err = 0.0;
	MatrixXd est = (KN_nq*W_q3 + src_nd*B_33).rowwise() + c_3.transpose();
	for (unsigned i=0; i < target_md.rows(); i+=1) {
		err += M_mn.row(i).dot((est.rowwise() - target_md.row(i)).rowwise().squaredNorm());
	}

	return err;
}

/** Computes the polar-decomposition cost. Uses rapprentice's fastrapp. */
double  RegOptProb::f_rotreg_cost(const VectorXd& x) {
	Map<Matrix3d> B_33((double*)x.data(),3,3);
	return RotReg(B_33, rot_coeff, scale_coeff);
}


/** Double-stochasticity of correspondence matrix M. */
void  RegOptProb::doubly_stochastic_constraints() {
	// doubly-stochastic M:
	const unsigned int m = m_vars.rows();
	const unsigned int n = m_vars.cols();

	for (unsigned i=0; i < m-1; i+=1) {
		AffExpr aff_cnt;
		aff_cnt.vars     = m_vars.row(i);
		aff_cnt.constant = -1.0;
		aff_cnt.coeffs   = vector<double>(n, 1.0);
		addLinearConstraint(aff_cnt, EQ);
	}

	for (unsigned i=0; i < n-1; i+=1) {
		AffExpr aff_cnt;
		aff_cnt.vars     = m_vars.col(i);
		aff_cnt.constant = -1.0;
		aff_cnt.coeffs   = vector<double>(m, 1.0);
		addLinearConstraint(aff_cnt, EQ);
	}
}
