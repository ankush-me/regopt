#include "problem_description.hpp"
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <boost/bind.hpp>

// John's sequential convex optimizer stuff
#include <sco/modeling_utils.hpp>
#include "rotreg.hpp"

using namespace Eigen;
using namespace std;
using namespace sco;


/** x is the big solution vector of the whole problem.
 *  vars are variables that index into the vector x
 *  this function extracts (from x) the values of the variables in vars. */
MatrixXd getMat(const VectorXd& x, const VarArray& vars) {
	MatrixXd out(vars.rows(), vars.cols());
	for (unsigned i=0; i < vars.rows(); i++) {
		for (unsigned j=0; j < vars.cols(); j++) {
			out(i,j) = x[vars(i,j).var_rep->index];
		}
	}
	return out;
}

MatrixXd getMat(const vector<double>& x, const VarArray& vars) {
	return getMat(VectorXd::Map(&x[0], x.size()), vars);
}


RegOptProb::RegOptProb(RegOptConfig::Ptr config) :
		src_nd(config->src_pts), target_md(config->target_pts),
		rot_coeff(config->rot_coeff), scale_coeff(config->scale_coeff),
		bend_coeff(config->bend_coeff),
		correspondence_coeff(config->correspondence_coeff),
		n_src(src_nd.rows()), m_target(target_md.rows()),
		rotreg(config->rotreg),
		K_nn(n_src, n_src) {
	init();
}


RegOptProb::RegOptProb(const MatrixX3d &src_pts, const MatrixX3d &target_pts,
		const Vector3d &rot_coeff_, double scale_coeff_, double bend_coeff_,
		double correspondence_coeff_, bool rotreg_)
: src_nd(src_pts), target_md(target_pts),
  rot_coeff(rot_coeff_), scale_coeff(scale_coeff_), bend_coeff(bend_coeff_),
  correspondence_coeff(correspondence_coeff_),
  n_src(src_nd.rows()), m_target(target_md.rows()),
  rotreg(rotreg_),
  K_nn(n_src, n_src) {

	init();
}

/** Does self-initialization. */
void RegOptProb::init() {
	// compute the tps-kernel matrix
	for (int i=0; i < n_src; i++)
		K_nn.row(i) = (src_nd.rowwise() - src_nd.row(i)).rowwise().norm();

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
 *    4. A (NX3)          : tps kernel weights
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


	// tps kernel weights
	last_size = getVars().size();
	vector<string> a_names(3*n_src);
	for (unsigned i=0; i < n_src; i+=1) {
		for(int j=0; j < 3; j+=1) {
			a_names[i*3 + j] = (boost::format("a_%i_%i")%i%j).str();
		}
	}
	createVariables(a_names);
	a_vars = VarArray(n_src, 3, getVars().data()+last_size);
}

void  RegOptProb::init_costs() {
	// add tps-cost
	ScalarOfVectorPtr f_ptr = ScalarOfVector::construct(boost::bind( &RegOptProb::f_tps_cost, this, _1 ));
	addCost(CostPtr(new CostFromFunc(f_ptr, getVars(), "f_tps_cost")));
}

/** Adds the following constraints:
 *  ===============================
 *    1. Doubly-stochastic constraint on M (allowing for outliers).
 *    2. (1 X).T * A = 0.
 *  M_ij \in [0,1] : this is already set in variable construction. */
void  RegOptProb::init_constraints() {
	doubly_stochastic_constraints();
	vanishing_moment_constraints();
}


/** Computes the tps-cost, given the current long solution vector x.
 *  Will use numerical differentiation for convex approximation.*/
double  RegOptProb::f_tps_cost(const VectorXd& x) {
	// retrieve current values in the correct matrix shapes.
	MatrixXd M_mn = getMat(x, m_vars.block(0,0, n_src, m_target));
	Vector3d c_3  = Vector3d(getMat(x, c_vars));
	MatrixXd B_33 = getMat(x, b_vars);
	MatrixXd A_n3 = getMat(x, a_vars);

	double objective = 0.0;

	// error term: sum_ij M_ij * || T_i - estimate_j ||^2
	double err = 0.0;
	MatrixXd est = (K_nn*A_n3 + src_nd*B_33).rowwise() + c_3;
	for (unsigned i=0; i < target_md.rows(); i+=1) {
		err += M_mn.row(i).dot((est.rowwise() - target_md.row(i)).rowwise().squaredNorm());
	}

	// correspondence term sum_ij M_ij:
	double M_sum = correspondence_coeff*M_mn.sum();

	// bending term: \lambda Tr(A.T*K*A)
	double bending = 0.0;
	MatrixXd KA = K_nn*A_n3;
	for (unsigned i=0; i < A_n3.cols(); i+=1) {
		bending += A_n3.col(i).dot(KA.col(i));
	}
	bending *= bend_coeff;

	objective = err + M_sum - bending;

	// add log-map cost
	if (rotreg)
		objective += f_rotreg_cost(B_33);

	return objective;
}

/** Computes the polar-decomposition cost. Uses rapprentice's fastrapp. */
double  RegOptProb::f_rotreg_cost(const MatrixXd &R) {
	return RotReg(R, rot_coeff, scale_coeff);
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

/** Constraints : (1 X).T * A = 0*/
void  RegOptProb::vanishing_moment_constraints() {

	// 1.T*A = 0
	for (unsigned i=0; i < a_vars.cols(); i+=1) {
		AffExpr aff_cnt;
		aff_cnt.vars     = a_vars.col(i);
		aff_cnt.constant = 0.0;
		aff_cnt.coeffs   = vector<double>(a_vars.rows(), 1.0);
		addLinearConstraint(aff_cnt, EQ);
	}

	// X.T*A = 0;  src_nd == X
	for(unsigned xi = 0; xi < src_nd.cols(); xi+=1) {
		for(unsigned ai = 0; ai < a_vars.cols(); ai +=1) {
			AffExpr aff_cnt;
			aff_cnt.vars     = a_vars.col(ai);
			aff_cnt.constant = 0.0;

			vector<double> coeffs(src_nd.rows());
			VectorXd::Map(&coeffs[0], coeffs.size()) = src_nd.col(xi);
			aff_cnt.coeffs   = coeffs;

			addLinearConstraint(aff_cnt, EQ);
		}
	}
}
