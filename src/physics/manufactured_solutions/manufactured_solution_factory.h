#ifndef __MANUFACTURED_SOLUTION_FACTORY__
#define __MANUFACTURED_SOLUTION_FACTORY__

#include "parameters/parameters_manufactured_convergence_study.h"
#include "manufactured_solution.h"

#include "sine.h"
#include "additive.h"
#include "cosine.h"
#include "arctangent.h"
#include "exponential.h"
#include "even_polynomial.h"
#include "polynomial.h"

namespace PHiLiP {
/// Create specified manufactured solution as ManufacturedSolutionFunction (base) object 
/** Factory design pattern whose job is to create the correct manufactured solution
 */
template <int dim, int nstate, typename real>
class ManufacturedSolutionFactory
{
public:
    /// Factory to return the correct manufactured solution given input file.
    static std::shared_ptr< ManufacturedSolutionFunction<dim,nstate,real> >
        create_ManufacturedSolution(const Parameters::ManufacturedConvergenceStudyParam::ManufacturedSolutionType manu_sol_type_input);
};


} // PHiLiP namespace

#endif