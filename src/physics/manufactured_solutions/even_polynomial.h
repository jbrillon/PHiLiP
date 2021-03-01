#ifndef __EVEN_POLYNOMIAL__
#define __EVEN_POLYNOMIAL__

#include <deal.II/base/tensor.h> // NOTE TO BE MODIFIED ACCORDINGLY

#include "parameters/parameters_manufactured_convergence_study.h"
#include "manufactured_solution.h"

namespace PHiLiP {
/// Additive manufactured solution function.  Derived from ManufacturedSolutionFunction.
template <int dim, typename real>
class Even_Polynomial : public ManufacturedSolutionFunction <dim, real>
{
protected:
    const double poly_max = 7;
public:
    /// Constructor
    Even_Polynomial ();

    /// Destructor
    ~Even_Polynomial () {};

    /// Manufactured solution exact value
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const;

    /// Gradient of the exact manufactured solution
    dealii::Tensor<1,dim,real> gradient (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const;

    /// Hessian of the exact manufactured solution
    dealii::SymmetricTensor<2,dim,real> hessian (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const;
};


} // PHiLiP namespace

#endif
