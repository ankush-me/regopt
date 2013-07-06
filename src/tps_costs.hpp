#pragma once

#include <Eigen/Core>
#include <vector>

// John's sequential convex optimizer stuff
#include <sco/modeling.hpp>
#include <utils/basic_array.hpp>
#include <utils/opt_utils.hpp>

// forward declarations
class RegOptProb;
class TPSOptProb;

/** Adds the residual cost [this is convex]. */
class ResidualCost: public sco::Cost {
public:
	ResidualCost(const TPSOptProb* prob);

	/** Return the value of sum_i ||y_i - f(x)_i||^2. */
	double value(const sco::DblVec& x);

	/** Returns an approximation to the weighted residual error.
	 *  Using Taylor's expansion. */
	sco::ConvexObjectivePtr convex(const sco::DblVec& x, sco::Model* model);

private:
	sco::QuadExpr cost_expr;
	Eigen::MatrixXd KN_nq;
	Eigen::MatrixXd X_n3,Y_n3;
	VarArray W,B,c;
	unsigned n_pts;
};


/** Adds the weighted residual cost. */
class WeightedResidualCost: public sco::Cost {
public:
	WeightedResidualCost(const RegOptProb* prob);

		/** Return the value of sum_ij m_ij ||y_j - f(x)_i||^2. */
	double value(const sco::DblVec& x);
	/** Returns an approximation to the weighted residual error.
	 *  Using Taylor's expansion. */
	sco::ConvexObjectivePtr convex(const sco::DblVec& x, sco::Model* model);

private:
	sco::QuadExpr lastApprox;
	Eigen::MatrixXd KN_nq;
	Eigen::MatrixXd X_n3,Y_m3;
	VarArray W,B,M,c;
	unsigned int n_src, m_target;

	/** Return the value of sum_ij m_ij ||y_j - f(x)_i||^2. */
	double comp_val(const Eigen::MatrixXd &M_mn, const Eigen::Vector3d &c_3,
			const Eigen::MatrixXd &B_33, const Eigen::MatrixXd W_q3);
};

/** Adds the -lambda Tr(A.T*K*A) cost [this is convex].
    -K is conditionally positive definite.*/
class BendingCost: public sco::Cost {
public:
	BendingCost(const Eigen::MatrixXd & K_nn_, const Eigen::MatrixXd &N_nq_, double bend_coeff_, const VarArray &a_vars_);
	double value(const sco::DblVec& x);
	sco::ConvexObjectivePtr convex(const sco::DblVec& x, sco::Model* model);
private:
	Eigen::MatrixXd NtKN_qq;
	double  bend_coeff;
	VarArray w_vars;
	sco::QuadExpr expr;
};

/** Correspondence term sum_ij M_ij [this is convex]. */
class CorrespondenceCost : public sco::Cost {
	sco::AffExpr  sum_expr;
public:
	CorrespondenceCost (double corr_coeff, const VarArray &m_vars);
	double value(const sco::DblVec& x);
	sco::ConvexObjectivePtr convex(const sco::DblVec& x, sco::Model* model);
};
