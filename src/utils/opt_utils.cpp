#include "opt_utils.hpp"

using namespace Eigen;
using namespace std;

/** Put the values in the proper place in the big solution vector X. */
void set_vals(vector<double> &x, const MatrixXd & data, const VarArray &index_mat) {
	assert(("Indexing array and data array should have same shape. ",
			data.cols()==index_mat.cols() && data.rows()==index_mat.rows()));

	for(unsigned i =0; i < data.rows(); i +=1) {
		for(unsigned j=0; j < data.cols(); j+=1) {
			x[index_mat(i,j).var_rep->index] = data(i,j);
		}
	}
}

/** Reverse of the above function. */
void extract_vals(const std::vector<double> &x, Eigen::MatrixXd &out, const VarArray &index_mat) {
	out = MatrixXd::Zero(index_mat.rows(), index_mat.cols());
	for(unsigned i =0; i < index_mat.rows(); i +=1) {
		for(unsigned j=0; j < index_mat.cols(); j+=1) {
			out(i,j) = x[index_mat(i,j).var_rep->index];
		}
	}
}
