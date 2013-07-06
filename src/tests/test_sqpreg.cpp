#include "sqpreg.hpp"
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

/**  A small example to show how to initialize a registration problem. */
int main(int c, char**v) {

	MatrixX3d src(4,3), target(10,3);
	src << 1,0,1, -1,2,2, 0,0,0, 3,3,3;

	RegOptConfig::Ptr config (new RegOptConfig);
	config->src_pts     = src;
	config->target_pts  = target;
	config->rot_coeff   = Vector3d(1,1,1);
	config->scale_coeff = 1;
	config->bend_coeff  = 1;
	config->correspondence_coeff = 1;
	config->rotreg      = true;

	pair<BasicTrustRegionSQPPtr, RegOptProb::Ptr> setup =  setup_reg_fit_optimization(config);
	RegOptProb::Ptr prob = setup.second;

	return 0;
}
