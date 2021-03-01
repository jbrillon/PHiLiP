#ifndef __EXPONENTIAL__
#define __EXPONENTIAL__

#include <deal.II/base/tensor.h> // NOTE TO BE MODIFIED ACCORDINGLY

#include "parameters/parameters_manufactured_convergence_study.h"
#include "manufactured_solution.h"

namespace PHiLiP {
/// Additive manufactured solution function.  Derived from ManufacturedSolutionFunction.
template <int dim, typename real>
class Exponential : public ManufacturedSolutionFunction <dim, real>
{
public:
    /// Constructor
    Exponential ();

    /// Destructor
    ~Exponential () {};

    /// Manufactured solution exact value
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const;

    /// Gradient of the exact manufactured solution
    dealii::Tensor<1,dim,real> gradient (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const;

    /// Hessian of the exact manufactured solution
    dealii::SymmetricTensor<2,dim,real> hessian (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const;
};


} // PHiLiP namespace

#endif
