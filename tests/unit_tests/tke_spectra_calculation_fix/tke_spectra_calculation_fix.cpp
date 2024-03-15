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

const double TOLERANCE = 1E-12;


int main (int argc, char * argv[])
{
    // MPI_Init(&argc, &argv);
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
    pcout << "cfl number: " << all_parameters.flow_solver_param.courant_friedrichs_lewy_number << std::endl;
    pcout << "expected KE at tf: " << all_parameters.flow_solver_param.expected_kinetic_energy_at_final_time << std::endl;

    AssertDimension(all_parameters.dimension, PHILIP_DIM);

    // const int max_dim = PHILIP_DIM;
    // const int max_nstate = 5;

    // if(all_parameters.run_type == PHiLiP::Parameters::AllParameters::RunType::flow_simulation) {
    std::unique_ptr<PHiLiP::FlowSolver::FlowSolver<dim,nstate>> flow_solver = PHiLiP::FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&all_parameters, parameter_handler);
        // std::unique_ptr<PHiLiP::FlowSolver::FlowSolverBase> flow_solver = PHiLiP::FlowSolver::FlowSolverFactory<dim,nstate>::create_flow_solver(&all_parameters,parameter_handler);
        // run_error = flow_solver->run();
        // pcout << "Flow simulation complete with run error code: " << run_error << std::endl;
    // }
    // static_cast<void>(flow_solver->run());
    // Compute kinetic energy and theoretical dissipation rate
    std::unique_ptr<PHiLiP::FlowSolver::PeriodicTurbulence<dim, nstate>> flow_solver_case = std::make_unique<PHiLiP::FlowSolver::PeriodicTurbulence<dim,nstate>>(&all_parameters);
    flow_solver_case->compute_and_update_integrated_quantities(*(flow_solver->dg));
    const double kinetic_energy_computed = flow_solver_case->get_integrated_kinetic_energy();
    pcout << "KE INTEGRATED = " << kinetic_energy_computed << std::endl;
    // const double theoretical_dissipation_rate_computed = flow_solver_case->get_vorticity_based_dissipation_rate();

    // const std::string input_filename_prefix = parameters_input->flow_solver_param.input_flow_setup_filename_prefix;
    // pcout << "reading values from file prefix  " << input_filename_prefix << " and projecting... " << std::flush;
    // SetInitialCondition<dim,nstate,double>::read_values_from_file_and_project(dg_input,input_filename_prefix);

    //const double ref_length = 1.0, mach_inf=1.0, angle_of_attack = 0.0, side_slip_angle = 0.0, gamma_gas = 1.4;
    //const double prandtl_number = 0.72, reynolds_number_inf=50000.0;
    const double a = 1.0 , b = 0.0, c = 1.4, d=0.72, e=1.0;
    PHiLiP::Physics::NavierStokes<dim, nstate, double> navier_stokes_physics = PHiLiP::Physics::NavierStokes<dim, nstate, double>(a,c,a,b,b,d,e,false,1.0);

    const double min = 0.0;
    const double max = 1.0;
    const int nx = 11;

    std::vector<unsigned int> repetitions(dim, nx);
    dealii::Point<dim,double> corner1, corner2;
    for (int d=0; d<dim; d++) { 
        corner1[d] = min;
        corner2[d] = max;
    }
    dealii::Triangulation<dim> grid;
    dealii::GridGenerator::subdivided_hyper_rectangle(grid, repetitions, corner1, corner2);

    std::array<double, nstate> conservative_soln;
    std::array<double, nstate> conservative_soln2;
    std::array<double, nstate> primitive_soln;
    for (auto cell : grid.active_cell_iterators()) {
        for (unsigned int v=0; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v) {
            const dealii::Point<dim,double> vertex = cell->vertex(v);
            for (int s=0; s<nstate; s++) {
                conservative_soln[s] = navier_stokes_physics.manufactured_solution_function->value(vertex, s);
            }
            primitive_soln = navier_stokes_physics.convert_conservative_to_primitive_templated(conservative_soln);
            conservative_soln2 = navier_stokes_physics.convert_primitive_to_conservative(primitive_soln);

            // Flipping back and forth between conservative and primitive solution result
            // in the same solution
            // assert_compare_array<nstate> ( conservative_soln, conservative_soln2, 1.0, TOLERANCE);
            // Manufactured solution gives positive density
            if(conservative_soln[0] < TOLERANCE) std::abort();
            // Manufactured solution gives positive energy
            if(conservative_soln[nstate-1] < TOLERANCE) std::abort();
            // Manufactured solution gives positive pressure
            if(primitive_soln[1+dim] < TOLERANCE) std::abort();

            if(navier_stokes_physics.compute_pressure(conservative_soln) < TOLERANCE) std::abort();
            if(navier_stokes_physics.compute_sound(conservative_soln) < TOLERANCE) std::abort();

        }
    }
    // std::cout<< "done"<<std::endl;
    // MPI_Finalize();
    return 0;
}

