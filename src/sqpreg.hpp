#pragma once

#include <boost/python.hpp>
#include "problem_description.hpp"
#include <sco/optimizers.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <utility>


namespace py = boost::python;

typedef boost::shared_ptr<sco::BasicTrustRegionSQP> BasicTrustRegionSQPPtr;

// put the values in the proper place in the big solution vector X.
void set_vals(std::vector<double> &x, const Eigen::MatrixXd & data, const VarArray &index_mat);

/** - Creates a registration problem.
 *    - adds it to the basic trust region solver
 *      - initializes the solution vector appropriately.
 *
 *   Returns a pointer to the sqp optimizer.*/
std::pair<BasicTrustRegionSQPPtr, RegOptProb::Ptr> setup_optimization(RegOptConfig::Ptr reg_config);
