#pragma once

#include "problem_description.hpp"
#include <sco/optimizers.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <utility>


typedef boost::shared_ptr<sco::BasicTrustRegionSQP> BasicTrustRegionSQPPtr;

/** - Creates a registration problem.
 *    - adds it to the basic trust region solver
 *      - initializes the solution vector appropriately.
 *
 *   Returns a pointer to the sqp optimizer.*/
std::pair<BasicTrustRegionSQPPtr, RegOptProb::Ptr> setup_optimization(RegOptConfig::Ptr reg_config);
