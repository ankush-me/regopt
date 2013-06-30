#include <Eigen/Core>
#include <iostream>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <boost/bind.hpp>


// John's sequential convex optimizer stuff
//#include <sco/expr_ops.hpp>
#include <sco/modeling.hpp>
#include <sco/modeling_utils.hpp>
#include <utils/basic_array.hpp>
//#include <sco/solver_interface.hpp>
//#include <trajopt/trajectory_costs.hpp>

#include "fastrapp.hpp"


using namespace Eigen;
using namespace std;
using namespace sco;

typedef Matrix< double, Dynamic, 3> MatrixX3d;
typedef util::BasicArray<Var> VarArray;


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


/** Class to hold data for the registration problem. **/
class RegOptProb : public OptProb {
public:

	RegOptProb(const MatrixX3d &src_pts, const MatrixX3d &target_pts,
			const Vector3d &rot_coeff_, double scale_coeff_, double bend_coeff_,
			double correspondence_coeff_, bool rotreg_=true)
	: src_nd(src_pts), target_md(target_pts),
	  rot_coeff(rot_coeff_), scale_coeff(scale_coeff_), bend_coeff(bend_coeff_),
	  correspondence_coeff(correspondence_coeff_),
	  n_src(src_nd.rows()), m_target(target_md.rows()),
	  rotreg(rotreg_),
	  K_nn(n_src, n_src) {

		for (int i=0; i < n_src; i++)
			K_nn.row(i) = (src_nd.rowwise() - src_nd.row(i)).rowwise().norm();

		init_vars();

	}

	~RegOptProb() {}

	MatrixX3d src_nd, target_md;
	Vector3d rot_coeff;
	double scale_coeff, bend_coeff;
	double correspondence_coeff;
	unsigned int n_src, m_target;
	bool rotreg;
	MatrixXd  K_nn;


	// hold the optimization variables for easy access
	VarArray m_vars;
	VarArray c_vars;
	VarArray b_vars;
	VarArray a_vars;


	typedef std::pair<string,string> StringPair;

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
	void init_vars() {
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


	/** Computes the tps-cost, given the current long solution vector x.
	 *  Will use numerical differentiation for convex approximation.*/
	double f_tps_cost(const VectorXd& x) {
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

		objective = err + M_sum + bending;

		// add log-map cost
		if (rotreg)
			objective += f_rotreg_cost(B_33);

		return objective;
	}


	/** Computes the polar-decomposition cost. Uses rapprentice's fastrapp. */
	double f_rotreg_cost(const MatrixXd &R) {
		return RotReg(R, rot_coeff, scale_coeff);
	}

	void init_costs() {
		// add tps-cost
		ScalarOfVectorPtr f_ptr = ScalarOfVector::construct(boost::bind( &RegOptProb::f_tps_cost, this, _1 ));
		addCost(CostPtr(new CostFromFunc(f_ptr, getVars(), "f_tps_cost")));
	}

	/** Adds the following constraints:
	 *  ===============================
	 *    1. Doubly-stochastic constraint on M (allowing for outliers).
	 *    2. (1 X).T * A = 0. */
	void init_constraints() {

	}

};


int main(int c, char**v) {

	MatrixX3d src(4,3), target(10,3);
	src << 0,0,0, 1,0,1, 2,2,2 , 3,3,3;

	RegOptProb prob(src, target, Vector3d(1,1,1), 1,1,1);

	cout << "m_vars : "<<endl;
	for (int i = 0; i < prob.m_vars.rows(); i+=1) {
		for (int j=0; j < prob.m_vars.cols(); j+=1) {
			cout << " " << prob.m_vars(i,j).var_rep->name;
		}
		cout <<endl;
	}

	cout << "a_vars : "<<endl;
	for (int i = 0; i < prob.a_vars.rows(); i+=1) {
		for (int j=0; j < prob.a_vars.cols(); j+=1) {
			cout << " " << prob.a_vars(i,j).var_rep->name;
		}
		cout <<endl;
	}

	return 0;
}
