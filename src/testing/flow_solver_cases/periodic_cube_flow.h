#ifndef __PERIODIC_CUBE_FLOW_H__
#define __PERIODIC_CUBE_FLOW_H__

// for FlowSolver class:
#include "physics/initial_conditions/initial_condition.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"
#include <deal.II/base/table_handler.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>
#include "flow_solver_case_base.h"
#include "physics/navier_stokes.h"

namespace PHiLiP {
namespace Tests {

#if PHILIP_DIM==1
    using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

template <int dim, int nstate>
class PeriodicCubeFlow : public FlowSolverCaseBase<dim,nstate>
{
public:
    /// Constructor.
    PeriodicCubeFlow(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~PeriodicCubeFlow() {};

    /// Function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;

protected:
    const int number_of_cells_per_direction; ///< Number of cells per direction for the grid
    const double domain_left; ///< Domain left-boundary value for generating the grid
    const double domain_right; ///< Domain right-boundary value for generating the grid
    const double domain_size; ///< Domain size (length in 1D, area in 2D, and volume in 3D)

    /// Display additional more specific flow case parameters
    virtual void display_additional_flow_case_specific_parameters() const override;

    /// Display grid parameters
    void display_grid_parameters() const;
};

template <int dim, int nstate>
class PeriodicTurbulence : public PeriodicCubeFlow<dim,nstate>
{
public:
    /// Constructor.
    PeriodicTurbulence(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~PeriodicTurbulence() {};
    
    /// Computes the integrated quantities over the domain simultaneously and updates the array storing them
    void compute_and_update_integrated_quantities(DGBase<dim, double> &dg);

    /** Gets the nondimensional integrated kinetic energy given a DG object from dg->solution
     *  -- Reference: Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                between the spectral difference and flux reconstruction schemes." 
     *                Computers & Fluids 221 (2021): 104922.
     * */
    double get_integrated_kinetic_energy() const;

    /** Gets the nondimensional integrated enstrophy given a DG object from dg->solution
     *  -- Reference: Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                between the spectral difference and flux reconstruction schemes." 
     *                Computers & Fluids 221 (2021): 104922.
     * */
    double get_integrated_enstrophy() const;

    /** Gets non-dimensional theoretical vorticity tensor based dissipation rate 
     *  Note: For incompressible flows or when dilatation effects are negligible 
     *  -- Reference: Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                between the spectral difference and flux reconstruction schemes." 
     *                Computers & Fluids 221 (2021): 104922.
     * */
    double get_vorticity_based_dissipation_rate() const;

    /** Evaluate non-dimensional theoretical pressure-dilatation dissipation rate
     *  -- Reference: Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                between the spectral difference and flux reconstruction schemes." 
     *                Computers & Fluids 221 (2021): 104922.
     * */
    double get_pressure_dilatation_based_dissipation_rate () const;

    /** Gets non-dimensional theoretical deviatoric strain-rate tensor based dissipation rate 
     *  -- Reference: Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                between the spectral difference and flux reconstruction schemes." 
     *                Computers & Fluids 221 (2021): 104922.
     * */
    double get_deviatoric_strain_rate_tensor_based_dissipation_rate() const;

protected:
    /// Filename (with extension) for the unsteady data table
    const std::string unsteady_data_table_filename_with_extension;

    /// Pointer to Navier-Stokes physics object for computing things on the fly
    std::shared_ptr< Physics::NavierStokes<dim,dim+2,double> > navier_stokes_physics;

    bool is_taylor_green_vortex = false; ///< Identifies if taylor green vortex case; initialized as false.
    bool is_viscous_flow = true; ///< Identifies if viscous flow; initialized as true.

    /// Display additional more specific flow case parameters
    void display_additional_flow_case_specific_parameters() const override;

    /// Function to compute the constant time step
    double get_constant_time_step(std::shared_ptr<DGBase<dim,double>> dg) const override;

    /// Compute the desired unsteady data and write it to a table
    void compute_unsteady_data_and_write_to_table(
            const unsigned int current_iteration,
            const double current_time,
            const std::shared_ptr <DGBase<dim, double>> dg,
            const std::shared_ptr<dealii::TableHandler> unsteady_data_table) override;

    /// List of possible integrated quantities over the domain
    enum IntegratedQuantitiesEnum {
        kinetic_energy,
        enstrophy,
        pressure_dilatation,
        deviatoric_strain_rate_tensor_magnitude_sqr,
        INTEGRATEDQUANTITIESENUM_NR_ITEMS // NOTE: This must be the last entry
    };
    /// Maximum number of computed quantities
    /*const*/ int MAX_NUMBER_OF_COMPUTED_QUANTITIES = IntegratedQuantitiesEnum::INTEGRATEDQUANTITIESENUM_NR_ITEMS;
    /// Array for storing the computed quantities; done for computational efficiency
    std::array<double,/*MAX_NUMBER_OF_COMPUTED_QUANTITIES*/4> integrated_quantities;
};

} // Tests namespace
} // PHiLiP namespace
#endif