#include <iostream>
#include "BayesOpt/AcquisitionStrategy"
#include "BayesOpt/BayesianOptimization"
#include "BayesOpt/GaussianProcess"
#include "BayesOpt/Kernel"

class Rosenbrock
{
 public:
  Rosenbrock(double a = 1.0, double b = 100.0) : a(a), b(b) {}

  double operator()(const Eigen::Vector2d& x) const
  {
    return std::pow(a - x[0], 2) + b * std::pow(x[1] - x[0] * x[0], 2);
  }

 private:
  double a, b;
};

int main(int argc, char** argv)
{
  RBFKernel kernel;
  GaussianProcess<RBFKernel> gp(kernel, 1e-5);
  LowerConfidenceBound<GaussianProcess<RBFKernel>> strategy;
  Rosenbrock func;
  std::vector<Eigen::Vector2d> bounds{{-2.0, 2.0}, {-2.0, 2.0}};
  BayesianOptimization<GaussianProcess<RBFKernel>, LowerConfidenceBound<GaussianProcess<RBFKernel>>>
      opt(gp, strategy, bounds, 100, 200);
  Eigen::Vector2d ans = opt.optimize<Rosenbrock>(func, true);
  std::cout << "Ans:" << ans.transpose() << std::endl;
  return 0;
}