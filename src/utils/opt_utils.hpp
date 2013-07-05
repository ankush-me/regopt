#pragma once

#include <Eigen/Core>
#include <vector>
#include <utils/basic_array.hpp>
#include <sco/modeling.hpp>
#include <limits>

typedef Eigen::Matrix< double, Eigen::Dynamic, 3> MatrixX3d;
typedef util::BasicArray<sco::Var> VarArray;

const double EPS = std::numeric_limits<double>::min();

/** Put the values in the proper place in the big solution vector X. */
void set_vals(std::vector<double> &x, const Eigen::MatrixXd & data, const VarArray &index_mat);

/** Reverse of the above function. */
void extract_vals(const std::vector<double> &x, Eigen::MatrixXd &out, const VarArray &index_mat);
