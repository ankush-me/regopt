

CostPtr xcost(new CostFromNumDiff());
prob->addCost(CostPtr(new Xcost(..., xvars, "xcost")));
prob->addCost(CostPtr(new Ycost(..., yvars, "xcost")));

VarVector xs = prob->createVariables(xnames);
VarVector ys = prob->createVariables(ynames);

struct Xcost : public Cost {
	double value();
	DblVector convex(...);
};


struct CostCalc : public ScalarOfVector {
	double value(const VectorXd& x);
};




//===================================================
void setupProblem(OptProbPtr& probptr, size_t nvars) {
	probptr.reset(new OptProb());
	vector<string> var_names;
	for (size_t i=0; i < nvars; ++i) {
		var_names.push_back( (boost::format("x_%i")%i).str() );
	}
	probptr->createVariables(var_names);
}

double f_QuadraticSeparable(const VectorXd& x) {
	return x(0)*x(0) + sq(x(1) - 1) + sq(x(2)-2);
}

void testQP(SQP, QuadraticSeparable)  {
	// if the problem is exactly a QP, it should be solved in one iteration
	OptProbPtr prob;
	setupProblem(prob, 3);
	prob->addCost(CostPtr(new CostFromFunc(ScalarOfVector::construct(&f_QuadraticSeparable), prob->getVars(), "f")));
	BasicTrustRegionSQP solver(prob);
	solver.trust_box_size_ = 100;
	vector<double> x = list_of(3)(4)(5);
	solver.initialize(x);
	OptStatus status = solver.optimize();
	ASSERT_EQ(status, OPT_CONVERGED);
	expectAllNear(solver.x(), list_of(0)(1)(2), 1e-3);
	// todo: checks on number of iterations and function evaluates
}


class AngVelCost: public Cost {
public:
	AngVelCost(vector<IncrementalRBPtr> rbs, const VarArray& r, double coeff) :
		rbs_(rbs), r_(r), coeff_(coeff) {
		name_="angvel";
	}

	double value(const DblVec& x) {
		MatrixXd q_(rbs_.size(),4);
		for (int i=0; i < rbs_.size(); ++i) q_.row(i) = toVector4d(rbs_[i]->m_q);
		MatrixXd rvals = getTraj(x, r_);
		MatrixXd qnew(q_.rows(), q_.cols());
		for (int i = 0; i < qnew.rows(); ++i) {
			qnew.row(i) = quatMult(quatExp(rvals.row(i)), q_.row(i));
		}
		MatrixXd wvals = getW(qnew, 1);
		return wvals.array().square().sum()*coeff_;
	}

	ConvexObjectivePtr convex(const DblVec& x, Model* model) {
		MatrixXd q_(rbs_.size(),4);
		for (int i=0; i < rbs_.size(); ++i) q_.row(i) = toVector4d(rbs_[i]->m_q);
		ConvexObjectivePtr out(new ConvexObjective(model));
		MatrixXd wvals = getW(q_, 1);
		for (int i = 0; i < wvals.rows(); ++i) {
			for (int j = 0; j < wvals.cols(); ++j) {
				out->addQuadExpr(exprMult(exprSquare(r_(i + 1, j) - r_(i, j) + wvals(i, j)), coeff_));
			}
		}
		return out;
	}
	vector<IncrementalRBPtr> rbs_;
	VarArray r_;
	double coeff_;
};

//===========================================================

/**
 * Holds all the data for a trajectory optimization problem
 * so you can modify it programmatically, e.g. add your own costs
 */
class TrajOptProb : public OptProb {
public:
  TrajOptProb();
  TrajOptProb(int n_steps, ConfigurationPtr rad);
  ~TrajOptProb() {}
  VarVector GetVarRow(int i) {
    return m_traj_vars.row(i);
  }
  Var& GetVar(int i, int j) {
    return m_traj_vars.at(i,j);
  }
  VarArray& GetVars() {
    return m_traj_vars;
  }
  int GetNumSteps() {return m_traj_vars.rows();}
  int GetNumDOF() {return m_traj_vars.cols();}
  ConfigurationPtr GetRAD() {return m_rad;}
  OR::EnvironmentBasePtr GetEnv() {return m_rad->GetEnv();}

  void SetInitTraj(const TrajArray& x) {m_init_traj = x;}
  TrajArray GetInitTraj() {return m_init_traj;}

  friend TrajOptProbPtr ConstructProblem(const ProblemConstructionInfo&);
private:
  VarArray m_traj_vars;
  ConfigurationPtr m_rad;
  TrajArray m_init_traj;
  typedef std::pair<string,string> StringPair;
};


TrajOptProb::TrajOptProb(int n_steps, ConfigurationPtr rad) : m_rad(rad) {
  DblVec lower, upper;
  m_rad->GetDOFLimits(lower, upper);
  int n_dof = m_rad->GetDOF();

  // put optimization joint limits a little inside robot joint limits
  // so numerical derivs work
  for (int i=0; i < n_dof; ++i) lower[i] += 1e-4;
  for (int i=0; i < n_dof; ++i) upper[i] -= 1e-4;

  vector<double> vlower, vupper;
  vector<string> names;
  for (int i=0; i < n_steps; ++i) {
    vlower.insert(vlower.end(), lower.data(), lower.data()+lower.size());
    vupper.insert(vupper.end(), upper.data(), upper.data()+upper.size());
    for (unsigned j=0; j < n_dof; ++j) {
      names.push_back( (boost::format("j_%i_%i")%i%j).str() );
    }
  }
  createVariables(names, vlower, vupper);
  m_traj_vars = VarArray(n_steps, n_dof, getVars().data());

}


TrajOptProb::TrajOptProb() {
}

//===========================================
// example of a convex cost

class JointPosCost : public Cost {
public:
  JointPosCost(const VarVector& vars, const VectorXd& vals, const VectorXd& coeffs);
  virtual ConvexObjectivePtr convex(const vector<double>& x, Model* model);
  virtual double value(const vector<double>&);
private:
  VarVector vars_;
  VectorXd vals_, coeffs_;
  QuadExpr expr_;
};

JointPosCost::JointPosCost(const VarVector& vars, const VectorXd& vals, const VectorXd& coeffs) :
    Cost("JointPos"), vars_(vars), vals_(vals), coeffs_(coeffs) {
    for (int i=0; i < vars.size(); ++i) {
      if (coeffs[i] > 0) {
        AffExpr diff = exprSub(AffExpr(vars[i]), AffExpr(vals[i]));
        exprInc(expr_, exprMult(exprSquare(diff), coeffs[i]));
      }
    }
}
double JointPosCost::value(const vector<double>& xvec) {
  VectorXd dofs = getVec(xvec, vars_);
  return ((dofs - vals_).array().square() * coeffs_.array()).sum();
}
ConvexObjectivePtr JointPosCost::convex(const vector<double>& x, Model* model) {
  ConvexObjectivePtr out(new ConvexObjective(model));
  out->addQuadExpr(expr_);
  return out;
}
//===========================================








