#include "rotreg.hpp"
#include <boost/foreach.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <iostream>
#include <vector>


using namespace Eigen;
using namespace std;


/////////////////
// http://en.wikipedia.org/wiki/Axis_angle#Log_map_from_SO.283.29_to_so.283.29
Vector3d LogMap(const Matrix3d& m) {
  double cosarg = (m.trace() - 1)/2;
  cosarg = fmin(cosarg, 1);
  cosarg = fmax(cosarg, -1);
  double theta = acos( cosarg );
  if (theta==0) return Vector3d::Zero();
  else return theta*(1/(2*sin(theta))) * Vector3d(m(2,1) - m(1,2), m(0,2)-m(2,0), m(1,0)-m(0,1));
}

double RotReg(const Matrix3d& b, const Vector3d& rot_coeffs, double scale_coeff) {
  // regularize rotation using polar decomposition
  JacobiSVD<Matrix3d> svd(b.transpose(), ComputeFullU | ComputeFullV);
  Vector3d s = svd.singularValues();
  if (b.determinant() <= 0) return INFINITY;
  return LogMap(svd.matrixU() * svd.matrixV().transpose()).cwiseAbs().dot(rot_coeffs) + s.array().log().square().sum()*scale_coeff;
}
