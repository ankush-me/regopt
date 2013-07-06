#pragma once

#include <sco/optimizers.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <utility>
#include <utils/opt_utils.hpp>

#include "tps_reg_fit_problem.hpp"
#include "tps_fit_problem.hpp"


/** - Creates a registration problem.
 *    - adds it to the basic trust region solver
 *      - initializes the solution vector appropriately.
 *
 *   Returns a pointer to the sqp optimizer.*/
std::pair<BasicTrustRegionSQPPtr, RegOptProb::Ptr> setup_reg_fit_optimization(RegOptConfig::Ptr reg_config);


/** - Creates a tps-fit problem.
 *    - adds it to the basic trust region solver
 *      - initializes the solution vector appropriately.
 *
 *   Returns a pointer to the sqp optimizer.*/
std::pair<BasicTrustRegionSQPPtr, TPSOptProb::Ptr> setup_fit_optimization(TPSOptConfig::Ptr reg_config);
