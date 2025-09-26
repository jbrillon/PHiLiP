#include <deal.II/base/utilities.h>
#include <assert.h>
#include <deal.II/grid/grid_generator.h>

// #include "assert_compare_array.h"
#include "parameters/all_parameters.h"
#include "physics/navier_stokes.h"
#include "physics/initial_conditions/set_initial_condition.h"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include "flow_solver/flow_solver_cases/periodic_turbulence.h"
#include <iostream>

#include "mesh/mesh_adaptation/mesh_adaptation.h"

const double TOLERANCE = 1E-12;


int main (int argc, char * argv[])
{
    const int dim = PHILIP_DIM;
    const int nstate = dim+2;

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    const int n_mpi = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    if (n_mpi==1 || mpi_rank==0) {
        dealii::deallog.depth_console(99);
    } else {
        dealii::deallog.depth_console(0);
    }
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);

    // Declare possible inputs
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);
    PHiLiP::Parameters::parse_command_line (argc, argv, parameter_handler);

    // Read inputs from parameter file and set those values in AllParameters object
    PHiLiP::Parameters::AllParameters all_parameters;
    pcout << "Reading input..." << std::endl;
    all_parameters.parse_parameters (parameter_handler);

    AssertDimension(all_parameters.dimension, PHILIP_DIM);

    /* IF READING IN THE SOLUTION, UNCOMMENT THIS BLOCK
    std::unique_ptr<PHiLiP::FlowSolver::FlowSolver<dim,nstate>> flow_solver = PHiLiP::FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&all_parameters, parameter_handler);
    */
    std::unique_ptr<PHiLiP::FlowSolver::FlowSolver<dim,nstate>> flow_solver = PHiLiP::FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&all_parameters,parameter_handler);
    int run_error = flow_solver->run();
    pcout << "Flow simulation complete with run error code: " << run_error << std::endl;
    /* IF READING IN THE SOLUTION, UNCOMMENT THIS BLOCK
        // (1) read in the first outputted velocity field
        // const std::string input_filename_prefix = "/media/julien/Samsung_T5/NarvalFiles/2023_JCP/robustness/viscous_TGV_ILES_NSFR_cDG_IR_2PF_GL_OI-0_dofs048_p5_procs64/setup_files/setup";
        const std::string input_filename_prefix = "./setup_files/setup";
        // const std::string input_filename_prefix = parameters_input->flow_solver_param.input_flow_setup_filename_prefix;
        // TO DO: In bash, cd into each run directory and call the executable; can give relative paths in this 
        // piece of code so that we dont need to modify it from case to case for the variable: input_filename_prefix
        // STEP 1: See if it works for the 48^3 first
        pcout << "reading values from file prefix: \n " << input_filename_prefix << " \n and projecting... " << std::flush;
        PHiLiP::SetInitialCondition<dim,nstate,double>::read_values_from_file_and_project(flow_solver->dg,input_filename_prefix);
        pcout << "done." << std::endl;
    */
    // create the PeriodicTurbulence object
    std::unique_ptr<PHiLiP::FlowSolver::PeriodicTurbulence<dim, nstate>> periodic_turbulence = std::make_unique<PHiLiP::FlowSolver::PeriodicTurbulence<dim,nstate>>(&all_parameters);
    // periodic_turbulence->output_velocity_field(flow_solver->dg,0,8.0);
    periodic_turbulence->output_velocity_field(flow_solver->dg,1,9.0); // output velocity field at original polynomial degree

    // (1) change poly_degree
    flow_solver->dg->set_p_degree_and_interpolate_solution(5);

    // (2) change mesh
    PHiLiP::Parameters::MeshAdaptationParam mesh_adaptation_param = (flow_solver->dg->all_parameters->mesh_adaptation_param);
    mesh_adaptation_param.refine_fraction = 0.0;
    mesh_adaptation_param.h_coarsen_fraction = 1.0;
    PHiLiP::MeshAdaptation<dim,double> mesh_adaptation(flow_solver->dg, &(mesh_adaptation_param));
    dealii::Vector<double> cellwise_errors (flow_solver->dg->high_order_grid->triangulation->n_active_cells())
    mesh_adaptation.cellwise_errors = cellwise_errors;
    // mesh_adaptation.adapt_mesh();
    mesh_adaptation.fixed_fraction_isotropic_refinement_and_coarsening();

    // do it again for the next outputted velocity field
    periodic_turbulence->output_velocity_field(flow_solver->dg,1,9.0);
    // const std::string input_filename_prefix = "/media/julien/Samsung_T5/NarvalFiles/2023_JCP/robustness/viscous_TGV_ILES_NSFR_cDG_IR_2PF_GL_OI-0_dofs048_p5_procs64/setup_files/setup";
    // pcout << "reading values from file prefix: \n " << input_filename_prefix << " \n and projecting... " << std::flush;
    // SetInitialCondition<dim,nstate,double>::read_values_from_file_and_project(flow_solver->dg,input_filename_prefix);
    // pcout << "done." << std::endl;
    return 0;
}

