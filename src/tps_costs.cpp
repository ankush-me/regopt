#include "tps_costs.hpp"
#include "tps_fit_problem.hpp"
#include "tps_reg_fit_problem.hpp"


#include <sco/expr_ops.hpp>

using namespace std;
using namespace Eigen;
using namespace sco;



/** Adds the weighted residual cost. */
ResidualCost::ResidualCost(const TPSOptProb* prob) :
		KN_nq(prob->KN_nq),
		X_n3(prob->src_nd), Y_n3(prob->target_nd),
		W(prob->w_vars),
		B(prob->b_vars),
		c(prob->c_vars),
		n_pts(prob->n_pts) {
	name_ = "residual_cost";

	for(int d=0; d < 3; d+=1) {// for x,y,z

		// add +c
		AffExpr c_aff;
		c_aff.vars = c.row(d);
		c_aff.coeffs = vector<double>(1,1.0);

		for(unsigned i=0; i < n_pts; i+=1) {// for each src-target pair
			// add +KN*w
			AffExpr w_aff;
			w_aff.vars = W.col(d);
			w_aff.coeffs.resize(w_aff.vars.size());
			VectorXd::Map(&w_aff.coeffs[0], w_aff.coeffs.size()) = KN_nq.row(i);

			// add +Xb
			AffExpr b_aff;
			b_aff.vars = B.col(d);
			b_aff.coeffs.resize(b_aff.vars.size());
			VectorXd::Map(&b_aff.coeffs[0], b_aff.coeffs.size()) = X_n3.row(i);

			AffExpr err_expr;
			exprInc(err_expr, c_aff);
			exprInc(err_expr, w_aff);
			exprInc(err_expr, b_aff);
			exprDec(err_expr, (double) Y_n3(i,d));
			QuadExpr err_sq = exprSquare(err_expr);

			exprInc(cost_expr, err_sq);
		}
	}
}

/** Return the value of sum_ij m_ij ||y_j - f(x)_i||^2. */
double ResidualCost::value(const DblVec& x) {
	// retrieve current values in the correct matrix shapes.
	return cost_expr.value(x);
}

/** Returns an approximation to the weighted residual error.
 *  Using Taylor's expansion. */
ConvexObjectivePtr ResidualCost::convex(const DblVec& x, Model* model) {
	ConvexObjectivePtr cvx_obj(new ConvexObjective(model));
	cvx_obj->addQuadExpr(cost_expr);
	return cvx_obj;
}


/** Adds the weighted residual cost. */
WeightedResidualCost::WeightedResidualCost(const RegOptProb* prob) :
										KN_nq(prob->KN_nq),
										X_n3(prob->src_nd), Y_m3(prob->target_md),
										W(prob->w_vars),
										B(prob->b_vars),
										M(prob->m_vars.block(0,0,  prob->m_target, prob->n_src)),
										c(prob->c_vars),
										n_src(prob->n_src), m_target(prob->m_target) {
	name_ = "w_residual_cost";
}


/** Return the value of sum_ij m_ij ||y_j - f(x)_i||^2. */
double WeightedResidualCost::comp_val(const MatrixXd &M_mn, const Vector3d &c_3, const MatrixXd &B_33, const MatrixXd W_q3) {
	double err = 0.0;
	MatrixXd est_n3 = (KN_nq*W_q3 + X_n3*B_33).rowwise() + c_3.transpose();
	for (unsigned i=0; i < M_mn.rows(); i+=1) {
		VectorXd err_i = (est_n3.rowwise() - Y_m3.row(i)).rowwise().squaredNorm();
		err += M_mn.row(i).dot(err_i);
	}
	return err;
}


/** Return the value of sum_ij m_ij ||y_j - f(x)_i||^2. */
double WeightedResidualCost::value(const DblVec& x) {
	// retrieve current values in the correct matrix shapes.
	MatrixXd M_mn = getMat(x, M);
	Vector3d c_3  = Vector3d(getMat(x, c));
	MatrixXd B_33 = getMat(x, B);
	MatrixXd W_q3 = getMat(x, W);

	return comp_val(M_mn, c_3, B_33, W_q3);
}


/** Returns an approximation to the weighted residual error.
 *  Using Taylor's expansion. */
ConvexObjectivePtr WeightedResidualCost::convex(const DblVec& x, Model* model) {
	// retrieve current values in the correct matrix shapes.
	MatrixXd M_mn = getMat(x, M);
	Vector3d c_3  = Vector3d(getMat(x, c));
	MatrixXd B_33 = getMat(x, B);
	MatrixXd W_q3 = getMat(x, W);

	// we index w.r.t the source points; Hence, need to
	// work with the transpose of M_mn.
	// sum_j c_ij \in R^n
	VectorXd sum_cij = M_mn.colwise().sum();
	VectorXd colsum  = sum_cij.array() + EPS;
	MatrixXd Mnorm_nm(M_mn.cols(), M_mn.rows());
	for(unsigned i=0; i < M_mn.cols(); i+=1)
		Mnorm_nm.row(i) = M_mn.col(i)/colsum[i];

	// sum_j c_norm_ij y_j \in R^3xn
	MatrixXd wy_3n  = (Mnorm_nm*Y_m3).transpose();

	// sum_j c_norm_ij y_j^2 \in R^3xn
	MatrixXd y2_n3 = Y_m3.array().square();
	MatrixXd wy2_3n = (Mnorm_nm*y2_n3).transpose();

	// final quadratic approximation
	QuadExpr out_expr;

	for(int d=0; d<3; d+=1) {// for x,y,z:
		// get the variables for the i-th dimension.
		AffExpr c_aff;
		c_aff.vars = c.row(d);
		c_aff.coeffs = vector<double>(1,1.0);

		AffExpr w_aff, b_aff;
		w_aff.vars = W.col(d);
		b_aff.vars = B.col(d);

		for(unsigned i=0; i < n_src; i+=1) {// for each source point


			// get coeffs for KN*w
			if (i==0)
				w_aff.coeffs.resize(w_aff.vars.size());
			VectorXd::Map(&w_aff.coeffs[0], w_aff.coeffs.size()) = KN_nq.row(i);

			// get coeffs for XB:
			if (i==0)
				b_aff.coeffs.resize(b_aff.vars.size());
			VectorXd::Map(&b_aff.coeffs[0], b_aff.coeffs.size()) = X_n3.row(i);

			AffExpr err;
			exprInc(err, c_aff);
			exprInc(err, b_aff);
			exprInc(err, w_aff);
			exprInc(err, (double) -wy_3n(d,i));
			QuadExpr err2 = exprSquare(err);
			exprInc(err2, (double) (wy2_3n(d,i) - (wy_3n(d,i)*wy_3n(d,i)) ) );
			exprScale(err2, (double) sum_cij[i]);

			exprInc(out_expr, err2);
		}
	}

	// now add terms for changes in c_ij:
	// --> calculate the estimate and the pairwise squared-error.
	MatrixXd est_n3 = (KN_nq*W_q3 + X_n3*B_33).rowwise() + c_3.transpose();
	MatrixXd err_mn(m_target, n_src);
	for (unsigned i=0; i < m_target; i+=1)
		err_mn.row(i) = (est_n3.rowwise() - Y_m3.row(i)).rowwise().squaredNorm();



	// set up the affine expression in C_ij : (c_ij' - c_ij_0)*err_ij^2
	AffExpr c_expr;
	c_expr.vars   = M.m_data;
	c_expr.coeffs.resize(m_target*n_src);
	MatrixXd::Map(&c_expr.coeffs[0], m_target, n_src) = err_mn;
	exprDec(c_expr, (err_mn.array()*M_mn.array()).sum());

	exprInc(out_expr, c_expr);
	ConvexObjectivePtr cvx_obj(new ConvexObjective(model));
	cvx_obj->addQuadExpr(out_expr);
	return cvx_obj;
}


/** Adds the -lambda Tr(A.T*K*A) cost.
    -K is conditionally positive definite.*/
BendingCost::BendingCost(const MatrixXd & KN_nq_, const MatrixXd &N_nq_, double bend_coeff_, const VarArray &w_vars_) :
														NtKN_qq(N_nq_.transpose()*KN_nq_), bend_coeff(bend_coeff_), w_vars(w_vars_) {
	name_ = "bending_cost";
	NtKN_qq *= -bend_coeff;

	assert (("Bending cost matrix shape mismatch.", NtKN_qq.rows()==w_vars.rows() && w_vars.cols()==3));
	// for each col of W : [Wx Wy Wz]
	for (int dim=0; dim < w_vars.cols(); dim+=1) {
		for(int i=0; i < NtKN_qq.rows(); i+=1) {
			QuadExpr qexpr;
			qexpr.vars1 = vector<Var>(w_vars.rows(), w_vars(i, dim));
			qexpr.vars2 = w_vars.col(dim);
			for(int j=0; j < NtKN_qq.cols(); j+=1)
				qexpr.coeffs.push_back(NtKN_qq(i,j));
			exprInc(expr, qexpr);
		}
	}
}


double BendingCost::value(const DblVec& x) {
	return	expr.value(x);
}

ConvexObjectivePtr BendingCost::convex(const DblVec& x, Model* model) {
	ConvexObjectivePtr cvx_obj(new ConvexObjective(model));
	cvx_obj->addQuadExpr(expr);
	return cvx_obj;
}

/** Correspondence term - a* sum_ij M_ij */
CorrespondenceCost::CorrespondenceCost (double corr_coeff, const VarArray &m_vars) {
	name_ = "corr_cost";
	sum_expr.vars = m_vars.m_data;
	sum_expr.coeffs = vector<double>(m_vars.size(), -corr_coeff);
}

double CorrespondenceCost::value(const sco::DblVec& x) {
	return sum_expr.value(x);
}

ConvexObjectivePtr CorrespondenceCost::convex(const sco::DblVec& x, sco::Model* model) {
	ConvexObjectivePtr cvx_obj(new ConvexObjective(model));
	cvx_obj->addAffExpr(sum_expr);
	return cvx_obj;
}
