#pragma once

#include <Eigen/Core>
#include <vector>
#include <utils/basic_array.hpp>
#include <sco/modeling.hpp>
#include <limits>

typedef boost::shared_ptr<sco::BasicTrustRegionSQP> BasicTrustRegionSQPPtr;
typedef Eigen::Matrix< double, Eigen::Dynamic, 3> MatrixX3d;
typedef util::BasicArray<sco::Var> VarArray;

const double EPS = std::numeric_limits<double>::min();

/** x is the big solution vector of the whole problem.
 *  vars are variables that index into the vector x
 *  this function extracts (from x) the values of the variables in vars. */
Eigen::MatrixXd getMat(const Eigen::VectorXd& x, const VarArray& vars);
Eigen::MatrixXd getMat(const std::vector<double>& x, const VarArray& vars);

/** Put the values in the proper place in the big solution vector X. */
void set_vals(std::vector<double> &x, const Eigen::MatrixXd & data, const VarArray &index_mat);

/** Reverse of the above function. */
void extract_vals(const std::vector<double> &x, Eigen::MatrixXd &out, const VarArray &index_mat);
