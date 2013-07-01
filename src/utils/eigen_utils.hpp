#pragma once
#include <Eigen/Core>


/** Returns the median : columnwise (axis=0) or rowwise (axis=1). */
Eigen::VectorXd median(const Eigen::MatrixXd &mat, int axis=0);
