#include "circular_shell_flow.h"

#include <stdlib.h>
#include <iostream>
#include "mesh/grids/circular_shell.hpp"

namespace PHiLiP {

namespace FlowSolver {
//=========================================================
// FLOW IN CIRCULAR SHELL DOMAIN
//=========================================================
template <int dim, int nstate>
CircularShellFlow<dim, nstate>::CircularShellFlow(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : FlowSolverCaseBase<dim, nstate>(parameters_input)
        , number_of_cells_in_radial_direction(this->all_param.flow_solver_param.number_of_cells_in_radial_direction)
        , domain_inner_radius(this->all_param.flow_solver_param.grid_inner_radius)
        , domain_outer_radius(this->all_param.flow_solver_param.grid_outer_radius)
{ }

template <int dim, int nstate>
std::shared_ptr<Triangulation> CircularShellFlow<dim,nstate>::generate_grid() const
{
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (
#if PHILIP_DIM!=1
            this->mpi_communicator
#endif
    );
    Grids::circular_shell<dim,Triangulation>(grid, domain_inner_radius, domain_outer_radius, number_of_cells_in_radial_direction);

    return grid;
}

template <int dim, int nstate>
void CircularShellFlow<dim,nstate>::display_grid_parameters() const
{
    const std::string grid_type_string = "circular_shell";
    // Display the information about the grid
    this->pcout << "- Grid type: " << grid_type_string << std::endl;
    this->pcout << "- - Grid degree: " << this->all_param.flow_solver_param.grid_degree << std::endl;
    this->pcout << "- - Domain dimensionality: " << dim << std::endl;
    this->pcout << "- - Domain inner radius: " << this->domain_inner_radius << std::endl;
    this->pcout << "- - Domain outer radius: " << this->domain_outer_radius << std::endl;
    this->pcout << "- - Number of cells in radial direction: " << this->number_of_cells_in_radial_direction << std::endl;
}

template <int dim, int nstate>
void CircularShellFlow<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    this->display_grid_parameters();
}

#if PHILIP_DIM==2
template class CircularShellFlow <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // FlowSolver namespace
} // PHiLiP namespace

