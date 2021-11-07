#ifndef __LARGE_EDDY_SIMULATION__
#define __LARGE_EDDY_SIMULATION__

#include "model.h"
#include "navier_stokes.h"
#include "euler.h"
#include "physics.h"

namespace PHiLiP {
namespace Physics {

/// Large Eddy Simulation equations. Derived from Navier-Stokes for modifying the stress tensor and heat flux, which is derived from PhysicsBase. 
template <int dim, int nstate, typename real>
class LargeEddySimulationBase : public ModelBase <dim, nstate, real>
{
public:
    /// Constructor
	LargeEddySimulationBase( 
	    const double                                              ref_length,
        const double                                              gamma_gas,
        const double                                              mach_inf,
        const double                                              angle_of_attack,
        const double                                              side_slip_angle,
        const double                                              prandtl_number,
        const double                                              reynolds_number_inf,
        const double                                              turbulent_prandtl_number);

    /// Navier-Stokes physics object
    NavierStokes<dim,dim+2,real>  navier_stokes_physics;

	/// Turbulent Prandtl number
	const double turbulent_prandtl_number;

    /// Convective flux: \f$ \mathbf{F}_{conv} \f$
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_flux (
        const std::array<real,nstate> &conservative_soln) const;

    /// Dissipative (i.e. viscous) flux: \f$ \mathbf{F}_{diss} \f$ 
    std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const;

    /// Source term for manufactured solution functions
    std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &solution) const;

    /// Nondimensionalized sub-grid scale (SGS) stress tensor, (tau^sgs)*
    virtual std::array<dealii::Tensor<1,dim,real>,dim> compute_SGS_stress_tensor (
        const std::array<real,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const = 0;

    /// Nondimensionalized sub-grid scale (SGS) heat flux, (q^sgs)*
    virtual dealii::Tensor<1,dim,real> compute_SGS_heat_flux (
        const std::array<real,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const = 0;

protected:
    /// Returns the square of the magnitude of the tensor
    template<typename real2> 
    real2 get_tensor_magnitude_sqr (const std::array<dealii::Tensor<1,dim,real2>,dim> &tensor) const;

    /// Templated dissipative (i.e. viscous) flux: \f$ \mathbf{F}_{diss} \f$ 
    template<typename real2>
    std::array<dealii::Tensor<1,dim,real2>,nstate> dissipative_flux_templated (
        const std::array<real2,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient) const;
};

/// Smagorinsky eddy viscosity model. Derived from Large Eddy Simulation.
template <int dim, int nstate, typename real>
class LargeEddySimulation_Smagorinsky : public LargeEddySimulationBase <dim, nstate, real>
{
public:
    /** Constructor for the sub-grid scale (SGS) model: Smagorinsky
     *  More details...
     *  Reference: To be put here
     */
    LargeEddySimulation_Smagorinsky( 
        const double                                              ref_length,
        const double                                              gamma_gas,
        const double                                              mach_inf,
        const double                                              angle_of_attack,
        const double                                              side_slip_angle,
        const double                                              prandtl_number,
        const double                                              reynolds_number_inf,
        const double                                              turbulent_prandtl_number,
        const double                                              model_constant,
        const double                                              grid_spacing);

    /// Nondimensionalized sub-grid scale (SGS) stress tensor, (tau^sgs)*
    std::array<dealii::Tensor<1,dim,real>,dim> compute_SGS_stress_tensor (
        const std::array<real,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const;

    /// Nondimensionalized sub-grid scale (SGS) heat flux, (q^sgs)* 
    std::array<dealii::Tensor<1,dim,real>,dim> compute_SGS_heat_flux (
        const std::array<real,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const;

    /// Eddy viscosity for the Smagorinsky model
    virtual real compute_eddy_viscosity(
        const std::array<real,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const;

protected:
    /// Templated nondimensionalized sub-grid scale (SGS) stress tensor, (tau^sgs)*
    template<typename real2> std::array<dealii::Tensor<1,dim,real2>,dim> compute_SGS_stress_tensor_templated (
        const std::array<real2,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const;

    /// Templated nondimensionalized sub-grid scale (SGS) heat flux, (q^sgs)*
    template<typename real2> 
    dealii::Tensor<1,dim,real2> compute_SGS_heat_flux_templated (
        const std::array<real2,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const;

    /// Templated eddy viscosity
    template<typename real2> real2 compute_eddy_viscosity_templated(
        const std::array<real2,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const;
};

/// WALE (Wall-Adapting Local Eddy-viscosity) eddy viscosity model. Derived from LargeEddySimulation_Smagorinsky for only modifying compute_eddy_viscosity.
template <int dim, int nstate, typename real>
class LargeEddySimulation_WALE : public LargeEddySimulation_Smagorinsky <dim, nstate, real>
{
public:
    /** Constructor for the sub-grid scale (SGS) model: Smagorinsky
     *  More details...
     *  Reference: Nicoud & Ducros (1999) "Subgrid-scale stress modelling based on the square of the velocity gradient tensor"
     */
    LargeEddySimulation_WALE( 
        const double                                    model_constant,
        std::shared_ptr< PhysicsBase<dim,dim+2,real> >  navier_stokes_physics_input,
        const double                                    turbulent_prandtl_number);

    /** Eddy viscosity for the WALE model. 
     *  Reference: Nicoud & Ducros (1999) "Subgrid-scale stress modelling based on the square of the velocity gradient tensor"
     */
    real compute_eddy_viscosity(
        const std::array<real,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const override;

protected:
    /// Templated eddy viscosity
    template<typename real2> real2 compute_eddy_viscosity_templated(
        const std::array<real2,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const;
};

} // Physics namespace
} // PHiLiP namespace

#endif