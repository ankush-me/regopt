#include "eigen_utils.hpp"
#include <Eigen/Core>

#include <algorithm>
#include <vector>

using namespace Eigen;
using namespace std;


/** Returns the median : columnwise (axis=0) or rowwise (axis=1). */
VectorXd median(const MatrixXd &mat, int axis) {
	MatrixXd copy_mat(mat);
	if (axis==1) copy_mat.transposeInPlace();

	VectorXd mvector(copy_mat.cols());

	for (unsigned i=0; i < mvector.size(); i+=1) {
		vector<double> col(copy_mat.rows());
		VectorXd::Map(&col[0], col.size()) = copy_mat.col(i);

		size_t mid_idx = col.size()/2;
		nth_element(col.begin(), col.begin() + mid_idx, col.end());
		mvector[i] = col[mid_idx];
	}
	return mvector;
}
