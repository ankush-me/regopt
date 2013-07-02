#pragma once

#include <Eigen/Core>
#include <iostream>
#include <vector>

// John's sequential convex optimizer stuff
#include <sco/modeling.hpp>
#include <utils/basic_array.hpp>
#include <utils/opt_utils.hpp>

/** Holds the input for a registration problem. **/
struct RegOptConfig {
	typedef boost::shared_ptr<RegOptConfig> Ptr;

	MatrixX3d src_pts, target_pts;
	Eigen::Vector3d rot_coeff;
	double scale_coeff, bend_coeff, correspondence_coeff;
	bool rotreg;
};

/** x is the big solution vector of the whole problem.
 *  vars are variables that index into the vector x
 *  this function extracts (from x) the values of the variables in vars. */
Eigen::MatrixXd getMat(const Eigen::VectorXd& x, const VarArray& vars);
Eigen::MatrixXd getMat(const std::vector<double>& x, const VarArray& vars);

/** Class to hold data for the registration problem. **/
class RegOptProb : public sco::OptProb {
public:

	typedef boost::shared_ptr<RegOptProb> Ptr;

	RegOptProb(RegOptConfig::Ptr reg_config);

	RegOptProb(const MatrixX3d &src_pts, const MatrixX3d &target_pts,
			const Eigen::Vector3d &rot_coeff_, double scale_coeff_, double bend_coeff_,
			double correspondence_coeff_, bool rotreg_=true);
	~RegOptProb() {}



	MatrixX3d src_nd, target_md;
	Eigen::Vector3d rot_coeff;
	double scale_coeff, bend_coeff;
	double correspondence_coeff;
	unsigned int n_src, m_target;
	bool rotreg;
	Eigen::MatrixXd  K_nn;

	// hold the optimization variables for easy access
	VarArray m_vars;
	VarArray c_vars;
	VarArray b_vars;
	VarArray a_vars;

private:

	/** Does self-initialization. */
	void init();


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
	void init_vars();
	void init_costs();

	/** Adds the following constraints:
	 *  ===============================
	 *    1. Doubly-stochastic constraint on M (allowing for outliers).
	 *    2. (1 X).T * A = 0.
	 *  M_ij \in [0,1] : this is already set in variable construction. */
	void init_constraints();

	/** Computes the tps-cost, given the current long solution vector x.
	 *  Will use numerical differentiation for convex approximation.*/
	double f_tps_cost(const Eigen::VectorXd& x);

	/** Computes the polar-decomposition cost. Uses rapprentice's fastrapp. */
	double f_rotreg_cost(const Eigen::MatrixXd &R);

	/** Double-stochasticity of correspondence matrix M. */
	void doubly_stochastic_constraints();

	/** Constraints : (1 X).T * A = 0*/
	void vanishing_moment_constraints();
};
