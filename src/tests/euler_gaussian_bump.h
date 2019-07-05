#ifndef __EULER_GAUSSIAN_BUMP_H__
#define __EULER_GAUSSIAN_BUMP_H__

#include <deal.II/grid/manifold_lib.h>

#include "tests.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

class BumpManifold: public dealii::ChartManifold<2,2,2> {
public:
    virtual dealii::Point<2>
    pull_back(const dealii::Point<2> &space_point) const;
    virtual dealii::Point<2>
    push_forward(const dealii::Point<2> &chart_point) const;
    
    virtual std::unique_ptr<dealii::Manifold<2,2> > clone() const;
};

/// Performs grid convergence for various polynomial degrees.
template <int dim, int nstate>
class EulerGaussianBump: public TestsBase
{
public:
    /// Constructor. Deleted the default constructor since it should not be used
    EulerGaussianBump () = delete;
    /// Constructor.
    /** Simply calls the TestsBase constructor to set its parameters = parameters_input
     */
    EulerGaussianBump(const Parameters::AllParameters *const parameters_input);

    ~EulerGaussianBump() {}; ///< Destructor.

    /// Grid convergence on Euler Gaussian Bump
    /** Will run the a grid convergence test for various p
     *  on multiple grids to determine the order of convergence.
     *
     *  Expecting the solution to converge at p+1. and output to converge at 2p+1.
     *  Note that the output solution currently convergens slightly suboptimally
     *  depending on the case (around 2p). The implementation of the boundary conditions
     *  play a large role on this adjoint consistency.
     *  
     *  Want to see entropy go to 0.
     */
    int run_test () const;

protected:

    void initialize_perturbed_solution(DGBase<dim,double> &dg, const Physics::PhysicsBase<dim,nstate,double> &physics) const;

    double integrate_entropy_over_domain(DGBase<dim,double> &dg) const;
};


//   /// Manufactured grid convergence
//   /** Currently the main function as all my test cases simply
//    *  check for optimal convergence of the solution
//    */
//   template<int dim>
//   int manufactured_grid_convergence (Parameters::AllParameters &parameters);

} // Tests namespace
} // PHiLiP namespace
#endif

