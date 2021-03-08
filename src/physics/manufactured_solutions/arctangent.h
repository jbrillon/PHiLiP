#ifndef __ARCTANGENT__
#define __ARCTANGENT__

#include <deal.II/base/tensor.h> // NOTE TO BE MODIFIED ACCORDINGLY

#include "parameters/parameters_manufactured_convergence_study.h"
#include "manufactured_solution.h"

namespace PHiLiP {
/// Additive manufactured solution function.  Derived from ManufacturedSolutionFunction.
template <int dim, int nstate, typename real>
class Arctangent : public ManufacturedSolutionFunction <dim, nstate, real>
{
protected:
    const double S1 = 50, S2 = -50;
    const double loc1 = 0.25, loc2 = 0.75;
    using dealii::Function<dim,real>::value;
    using dealii::Function<dim,real>::gradient;
    using dealii::Function<dim,real>::hessian;
public:
    /// Constructor
    Arctangent () {};

    /// Destructor
    ~Arctangent () {};

    /// Manufactured solution exact value
    real value (const dealii::Point<dim,real> &point, const int istate = 0) const;

    /// Gradient of the exact manufactured solution
    dealii::Tensor<1,dim,real> gradient (const dealii::Point<dim,real> &point, const int istate = 0) const;

    /// Hessian of the exact manufactured solution
    dealii::SymmetricTensor<2,dim,real> hessian (const dealii::Point<dim,real> &point, const int istate = 0) const;

// protected:

};


} // PHiLiP namespace

#endif
