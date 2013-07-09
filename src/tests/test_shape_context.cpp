#include "sqpregpy/pyshape_context.hpp"
#include <Eigen/Core>
#include <iostream>

using namespace std;
using namespace Eigen;


/**  A small example to show how to initialize a registration problem. */
int main(int c, char**v) {

	MatrixX3d pts = MatrixXd::Random(10,3);
	MatrixXd sc   = shape_context(pts);
	cout << " shape context has size : ("<< sc.rows()<<", "<<sc.cols()<<")"<<endl;

	MatrixXd d = shape_distance(sc, sc);
	cout << "shape distance [should be all zeros] : " <<endl<<d<<endl;

	return 0;
}
