#ifndef __POLYNOMIAL__
#define __POLYNOMIAL__

#include <deal.II/base/tensor.h> // NOTE TO BE MODIFIED ACCORDINGLY

#include "parameters/parameters_manufactured_convergence_study.h"
#include "manufactured_solution.h"

namespace PHiLiP {
/// Additive manufactured solution function.  Derived from ManufacturedSolutionFunction.
template <int dim, int nstate, typename real>
class Polynomial : public ManufacturedSolutionFunction <dim, nstate, real>
{
protected:
    using dealii::Function<dim,real>::value;
    using dealii::Function<dim,real>::gradient;
    using dealii::Function<dim,real>::hessian;
    using ManufacturedSolutionFunction<dim,nstate,real>::base_values;
public:
    /// Constructor
    Polynomial ();

    /// Destructor
    ~Polynomial () {};

    /// Manufactured solution exact value
    real value (const dealii::Point<dim,real> &point, const int istate = 0) const;

    /// Gradient of the exact manufactured solution
    dealii::Tensor<1,dim,real> gradient (const dealii::Point<dim,real> &point, const int istate = 0) const;

    /// Hessian of the exact manufactured solution
    dealii::SymmetricTensor<2,dim,real> hessian (const dealii::Point<dim,real> &point, const int istate = 0) const;
};


} // PHiLiP namespace

#endif
