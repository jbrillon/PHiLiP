#include <cmath>
#include <vector>

#include "ADTypes.hpp"

#include "model.h"

namespace PHiLiP {
namespace Physics {

//================================================================
// Models Base Class
//================================================================
template <int dim, int nstate, typename real>
ModelBase<dim, nstate, real>::ModelBase(
    const dealii::Tensor<2,3,double>                          /*input_diffusion_tensor*/,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function_input):
        manufactured_solution_function(manufactured_solution_function_input)
{
    // Nothing to do here so far
}
//----------------------------------------------------------------

//----------------------------------------------------------------
//----------------------------------------------------------------
//----------------------------------------------------------------
// Instantiate explicitly
template class ModelBase < PHILIP_DIM, PHILIP_DIM+2, double >;
template class ModelBase < PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class ModelBase < PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class ModelBase < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class ModelBase < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

} // Physics namespace
} // PHiLiP namespace