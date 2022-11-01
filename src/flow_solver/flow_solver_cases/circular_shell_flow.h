#ifndef __CIRCULAR_SHELL_FLOW_H__
#define __CIRCULAR_SHELL_FLOW_H__

#include "flow_solver_case_base.h"

namespace PHiLiP {
namespace FlowSolver {

#if PHILIP_DIM==1
using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

template <int dim, int nstate>
class CircularShellFlow : public FlowSolverCaseBase<dim,nstate>
{
public:
    /// Constructor.
    CircularShellFlow(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~CircularShellFlow() {};

    /// Function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;

protected:
    const int number_of_cells_in_radial_direction; ///< Number of cells in radial direction for the grid
    const double domain_inner_radius; ///< Domain inner-radius value for generating the grid
    const double domain_outer_radius; ///< Domain outer-radius value for generating the grid

    /// Display additional more specific flow case parameters
    virtual void display_additional_flow_case_specific_parameters() const override;

    /// Display grid parameters
    void display_grid_parameters() const;
};

} // FlowSolver namespace
} // PHiLiP namespace
#endif
