#pragma once

#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/format.hpp>

// John's sequential convex optimizer stuff
#include <sco/modeling.hpp>
#include <utils/basic_array.hpp>
#include <utils/opt_utils.hpp>


/** Holds the input for a registration problem. **/
struct TPSOptConfig {
	typedef boost::shared_ptr<TPSOptConfig> Ptr;

	MatrixX3d src_pts, target_pts;
	Eigen::VectorXd  weights;
	Eigen::Vector3d rot_coeff;
	double scale_coeff, bend_coeff;
	bool rotreg;
};

/** Class to hold data for the registration problem. **/
class TPSOptProb : public sco::OptProb {
public:

	typedef boost::shared_ptr<TPSOptProb> Ptr;

	TPSOptProb(TPSOptConfig::Ptr reg_config);

	TPSOptProb(const MatrixX3d &src_pts, const MatrixX3d &target_pts,
			const Eigen::VectorXd &weights,
			const Eigen::Vector3d &rot_coeff_, double scale_coeff_, double bend_coeff_,
			bool rotreg_=true);
	~TPSOptProb() {}


	MatrixX3d src_nd, target_nd;
	Eigen::VectorXd w_n;
	unsigned int n_pts;
	Eigen::Vector3d rot_coeff;
	double scale_coeff, bend_coeff;
	bool rotreg;
	Eigen::MatrixXd  KN_nq;
	Eigen::MatrixXd  N_nq; // null of [X 1]
	unsigned call_count;

	// hold the optimization variables for easy access
	VarArray c_vars;
	VarArray b_vars;
	VarArray w_vars;

private:

	/** Does self-initialization. */
	void init();

	/** Initialize the optimization variables.
	 *
	 *  say N = n_src, M = m_target
	 *
	 *  The variables are:
	 *  ==================
	 *    1. c (3x1)          : translation
	 *    2. B (3x3)          : affine part
	 *    3. A (NX3)          : tps kernel weights  */
	void init_vars();

	void init_costs();

	/** Adds the following constraints:
	 *  ===============================*/
	 void init_constraints();

	/** Computes the polar-decomposition cost. Uses rapprentice's fastrapp. */
	double f_rotreg_cost(const Eigen::VectorXd& x);
};
