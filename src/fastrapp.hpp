#pragma once
#include <Eigen/Core>

/////////////////
// http://en.wikipedia.org/wiki/Axis_angle#Log_map_from_SO.283.29_to_so.283.29
Eigen::Vector3d LogMap(const Eigen::Matrix3d& m);
double RotReg(const Eigen::Matrix3d& b, const Eigen::Vector3d& rot_coeffs, double scale_coeff);
