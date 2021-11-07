#include "parameters/all_parameters.h"
#include "parameters/parameters_manufactured_solution.h"

#include <deal.II/base/tensor.h>

#include "ADTypes.hpp"

#include "physics_factory.h"
#include "manufactured_solution.h"
#include "physics.h"
#include "convection_diffusion.h"
#include "burgers.h"
#include "burgers_rewienski.h"
#include "euler.h"
#include "mhd.h"
#include "navier_stokes.h"
#include "physics_model.h"

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
std::shared_ptr < PhysicsBase<dim,nstate,real> >
PhysicsFactory<dim,nstate,real>
::create_Physics(const Parameters::AllParameters *const parameters_input)
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    PDE_enum pde_type = parameters_input->pde_type;

    return create_Physics(parameters_input, pde_type);
}

template <int dim, int nstate, typename real>
std::shared_ptr < PhysicsBase<dim,nstate,real> >
PhysicsFactory<dim,nstate,real>
::create_Physics(const Parameters::AllParameters *const parameters_input,
                 const Parameters::AllParameters::PartialDifferentialEquation pde_type)
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;

    // generating the manufactured solution from the manufactured solution factory
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> >  manufactured_solution_function 
        = ManufacturedSolutionFactory<dim,real>::create_ManufacturedSolution(parameters_input, nstate);

    // setting the diffusion tensor and advection vectors from parameters (if needed)
    const dealii::Tensor<2,3,double> diffusion_tensor      = parameters_input->manufactured_convergence_study_param.manufactured_solution_param.diffusion_tensor;
    const dealii::Tensor<1,3,double> advection_vector      = parameters_input->manufactured_convergence_study_param.manufactured_solution_param.advection_vector;
    const double                     diffusion_coefficient = parameters_input->manufactured_convergence_study_param.manufactured_solution_param.diffusion_coefficient;

    if (pde_type == PDE_enum::advection || pde_type == PDE_enum::advection_vector) {
        if constexpr (nstate<=2) 
            return std::make_shared < ConvectionDiffusion<dim,nstate,real> >(
                true, false,
                diffusion_tensor, advection_vector, diffusion_coefficient,
                manufactured_solution_function);
    } else if (pde_type == PDE_enum::diffusion) {
        if constexpr (nstate==1) 
            return std::make_shared < ConvectionDiffusion<dim,nstate,real> >(
                false, true,
                diffusion_tensor, advection_vector, diffusion_coefficient,
                manufactured_solution_function);
    } else if (pde_type == PDE_enum::convection_diffusion) {
        if constexpr (nstate==1) 
            return std::make_shared < ConvectionDiffusion<dim,nstate,real> >(
                true, true,
                diffusion_tensor, advection_vector, diffusion_coefficient,
                manufactured_solution_function);
    } else if (pde_type == PDE_enum::burgers_inviscid) {
        if constexpr (nstate==dim) 
            return std::make_shared < Burgers<dim,nstate,real> >(
                parameters_input->burgers_param.diffusion_coefficient,
                true, false,
                diffusion_tensor, 
                manufactured_solution_function);
    } else if (pde_type == PDE_enum::burgers_viscous) {
        if constexpr (nstate==dim)
            return std::make_shared < Burgers<dim,nstate,real> >(
                    parameters_input->burgers_param.diffusion_coefficient,
                    true, true,
                    diffusion_tensor,
                    manufactured_solution_function);
    } else if (pde_type == PDE_enum::burgers_rewienski) {
        if constexpr (nstate==dim)
            return std::make_shared < BurgersRewienski<dim,nstate,real> >(
                    parameters_input->burgers_param.rewienski_a,
                    parameters_input->burgers_param.rewienski_b,
                    parameters_input->burgers_param.rewienski_manufactured_solution,
                    true,
                    false,
                    diffusion_tensor,
                    manufactured_solution_function);
    } else if (pde_type == PDE_enum::euler) {
        if constexpr (nstate==dim+2) {
            return std::make_shared < Euler<dim,nstate,real> > (
                parameters_input->euler_param.ref_length,
                parameters_input->euler_param.gamma_gas,
                parameters_input->euler_param.mach_inf,
                parameters_input->euler_param.angle_of_attack,
                parameters_input->euler_param.side_slip_angle,
                diffusion_tensor, 
                manufactured_solution_function);
        }
    } else if (pde_type == PDE_enum::mhd) {
        if constexpr (nstate == 8) 
            return std::make_shared < MHD<dim,nstate,real> > (
                parameters_input->euler_param.gamma_gas,
                diffusion_tensor, 
                manufactured_solution_function);
    } else if (pde_type == PDE_enum::navier_stokes) {
        if constexpr (nstate==dim+2) {
            return std::make_shared < NavierStokes<dim,nstate,real> > (
                parameters_input->euler_param.ref_length,
                parameters_input->euler_param.gamma_gas,
                parameters_input->euler_param.mach_inf,
                parameters_input->euler_param.angle_of_attack,
                parameters_input->euler_param.side_slip_angle,
                parameters_input->navier_stokes_param.prandtl_number,
                parameters_input->navier_stokes_param.reynolds_number_inf,
                diffusion_tensor, 
                manufactured_solution_function);
        }
    } else if (pde_type == PDE_enum::physics_model) {
        if constexpr (nstate>=dim+2) {
            return create_Physics_Model(parameters_input, 
                                        diffusion_tensor, 
                                        manufactured_solution_function);
        }
    } else {
        // prevent warnings for dim=3,nstate=4, etc.
        (void) diffusion_tensor;
        (void) advection_vector;
        (void) diffusion_coefficient;
    }
    std::cout << "Can't create PhysicsBase, invalid PDE type: " << pde_type << std::endl;
    assert(0==1 && "Can't create PhysicsBase, invalid PDE type");
    return nullptr;
}

template <int dim, int nstate, typename real>
std::shared_ptr < PhysicsBase<dim,nstate,real> >
PhysicsFactory<dim,nstate,real>
::create_Physics_Model(const Parameters::AllParameters                           *const parameters_input,
                       const dealii::Tensor<2,3,double>                          diffusion_tensor,
                       std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function)
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;

    using Model_enum = Parameters::AllParameters::ModelType;
    Model_enum model_type = parameters_input->model_type;
    
    // ===============================================================================
    // Physics Model
    // ===============================================================================
    
    // Create baseline physics object
    PDE_enum baseline_physics_type;

    // Number of states in the baseline physics
    int nstate_baseline_physics;
    
    // model object for the additional terms and equations to the baseline physics  
    // TO DO: make this an input to create_Physics
    // std::unique_ptr< ModelBase<dim,nstate,real> > model;

    // -------------------------------------------------------------------------------
    // Large Eddy Simulation (LES)
    // -------------------------------------------------------------------------------
    if (model_type == Model_enum::large_eddy_simulation) {
        // Assign baseline physics type (and corresponding nstates) based on the physics model type
        // -- Assign nstates for the baseline physics
        nstate_baseline_physics = dim+2;
        // -- Assign baseline physics type
        if(parameters_input->physics_model_param.euler_turbulence) {
            baseline_physics_type = PDE_enum::euler;
        }
        else {
            baseline_physics_type = PDE_enum::navier_stokes;
        }

        // Create Large Eddy Simulation (LES) model based on the SGS model type
        // using SGS_enum = Parameters::PhysicsModelParam::SubGridScaleModel;
        // SGS_enum sgs_model_type = parameters_input->physics_model_param.SGS_model_type;
        // if (sgs_model_type == SGS_enum::smagorinsky) {
        //     // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        //     // Smagorinsky model
        //     // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        //     model = std::make_shared < LargeEddySimulation_Smagorinsky<dim,nstate,real> > (
        //         navier_stokes_physics,
        //         parameters_input->physics_model_param.turbulent_prandtl_number,
        //         parameters_input->physics_model_param.smagorinsky_model_constant);
        // } else if (sgs_model_type == SGS_enum::wall_adaptive_local_eddy_viscosity) {
        //     // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        //     // WALE (Wall-Adapting Local Eddy-viscosity) eddy viscosity model
        //     // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        //     model = std::make_shared < LargeEddySimulation_WALE<dim,nstate,real> > (
        //         navier_stokes_physics,
        //         parameters_input->physics_model_param.turbulent_prandtl_number,
        //         parameters_input->physics_model_param.WALE_model_constant);
        // }
        // model = nullptr;

        // Create the physics model object in physics
        if constexpr (nstate==dim+2) {
            return std::make_shared < PhysicsModel<dim,nstate,real> > (
                    baseline_physics_type,
                    nstate_baseline_physics,
                    /*nullptr,*/
                    diffusion_tensor, 
                    manufactured_solution_function);   
        }
    } else {
        // prevent warnings for dim=3,nstate=4, etc.
        (void) diffusion_tensor;
        (void) baseline_physics_type;
        (void) nstate_baseline_physics;
    }    
    std::cout << "Can't create PhysicsModel, invalid ModelType type: " << model_type << std::endl;
    assert(0==1 && "Can't create PhysicsModel, invalid ModelType type");
    return nullptr;
}

template class PhysicsFactory<PHILIP_DIM, 1, double>;
template class PhysicsFactory<PHILIP_DIM, 2, double>;
template class PhysicsFactory<PHILIP_DIM, 3, double>;
template class PhysicsFactory<PHILIP_DIM, 4, double>;
template class PhysicsFactory<PHILIP_DIM, 5, double>;
template class PhysicsFactory<PHILIP_DIM, 8, double>;

template class PhysicsFactory<PHILIP_DIM, 1, FadType >;
template class PhysicsFactory<PHILIP_DIM, 2, FadType >;
template class PhysicsFactory<PHILIP_DIM, 3, FadType >;
template class PhysicsFactory<PHILIP_DIM, 4, FadType >;
template class PhysicsFactory<PHILIP_DIM, 5, FadType >;
template class PhysicsFactory<PHILIP_DIM, 8, FadType >;

template class PhysicsFactory<PHILIP_DIM, 1, RadType >;
template class PhysicsFactory<PHILIP_DIM, 2, RadType >;
template class PhysicsFactory<PHILIP_DIM, 3, RadType >;
template class PhysicsFactory<PHILIP_DIM, 4, RadType >;
template class PhysicsFactory<PHILIP_DIM, 5, RadType >;
template class PhysicsFactory<PHILIP_DIM, 8, RadType >;

template class PhysicsFactory<PHILIP_DIM, 1, FadFadType >;
template class PhysicsFactory<PHILIP_DIM, 2, FadFadType >;
template class PhysicsFactory<PHILIP_DIM, 3, FadFadType >;
template class PhysicsFactory<PHILIP_DIM, 4, FadFadType >;
template class PhysicsFactory<PHILIP_DIM, 5, FadFadType >;
template class PhysicsFactory<PHILIP_DIM, 8, FadFadType >;

template class PhysicsFactory<PHILIP_DIM, 1, RadFadType >;
template class PhysicsFactory<PHILIP_DIM, 2, RadFadType >;
template class PhysicsFactory<PHILIP_DIM, 3, RadFadType >;
template class PhysicsFactory<PHILIP_DIM, 4, RadFadType >;
template class PhysicsFactory<PHILIP_DIM, 5, RadFadType >;
template class PhysicsFactory<PHILIP_DIM, 8, RadFadType >;



} // Physics namespace
} // PHiLiP namespace

