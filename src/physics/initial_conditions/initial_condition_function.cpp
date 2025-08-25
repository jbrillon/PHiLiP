#include <deal.II/base/function.h>
#include "initial_condition_function.h"
// For initial conditions which need to refer to physics
#include "physics/physics_factory.h"

namespace PHiLiP {

// =========================================================
// Initial Condition Base Class
// =========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction<dim,nstate,real>
::InitialConditionFunction ()
    : dealii::Function<dim,real>(nstate)//,0.0) // 0.0 denotes initial time (t=0)
{
    // Nothing to do here yet
}

// ========================================================
// Turbulent Channel Flow -- Initial Condition (Laminar x-velocity)
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_TurbulentChannelFlow<dim,nstate,real>
::InitialConditionFunction_TurbulentChannelFlow (
    const Physics::NavierStokes<dim,nstate,double> navier_stokes_physics_,
    const double channel_friction_velocity_reynolds_number_,
    const double domain_length_x_,
    const double domain_length_y_,
    const double domain_length_z_)
    : InitialConditionFunction<dim,nstate,real>()
    , navier_stokes_physics(navier_stokes_physics_)
    , channel_friction_velocity_reynolds_number(channel_friction_velocity_reynolds_number_)
    , domain_length_x(domain_length_x_)
    , domain_length_y(domain_length_y_)
    , domain_length_z(domain_length_z_)
    , channel_height(domain_length_y)
    , half_channel_height(0.5*channel_height)
{}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_TurbulentChannelFlow<dim, nstate, real>
::get_distance_from_wall(const dealii::Point<dim,real> &point) const
{
    // Get closest wall normal distance
    real y = point[1]; // y-coordinate of position
    real dist_from_wall = half_channel_height; // represents distance normal to top/bottom wall (which ever is closer); y-domain bounds are [-half_channel_height, half_channel_height]
    if(y > 0.0){
        dist_from_wall -= y; // distance from top wall
    } else if(y < 0.0) {
        dist_from_wall += y; // distance from bottom wall
    }
    return dist_from_wall;
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_TurbulentChannelFlow<dim, nstate, real>
::x_velocity(const dealii::Point<dim,real> &point, const real /*density*/, const real /*temperature*/) const
{
    // Laminar velocity profile
    // Reference: G. LODATO, P. CASTONGUAY AND A. JAMESON, "Discrete filter operators for large-eddy simulation using high-order spectral difference methods", Int. J. Numer. Meth. Fluids (2012)
    const real x_velocity = (15.0/8.0)*pow(1.0-pow(point[1]/half_channel_height,2.0),2.0);
    return x_velocity;
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_TurbulentChannelFlow<dim, nstate, real>
::y_velocity(const dealii::Point<dim,real> &point) const
{
    // Setup perturbed velocity
    const real C = 0.1; // Reference: G. LODATO, P. CASTONGUAY AND A. JAMESON, "Discrete filter operators for large-eddy simulation using high-order spectral difference methods", Int. J. Numer. Meth. Fluids (2012)
    const real x_loc = 0.0; // x-point at which to center the disturbance <-- Reference: R. Rossi / Journal of Computational Physics 228 (2009) 1639–1657
    const real y_loc = 0.0; // y-point at which to center the disturbance <-- Reference: R. Rossi / Journal of Computational Physics 228 (2009) 1639–1657
    const real pi_val = 3.141592653589793238;
    const real beta = 4.0*pi_val; // Reference: G. LODATO, P. CASTONGUAY AND A. JAMESON, "Discrete filter operators for large-eddy simulation using high-order spectral difference methods", Int. J. Numer. Meth. Fluids (2012)
    const real x_scale = domain_length_x;
    const real y_scale = domain_length_y;
    const real z_scale = domain_length_z;
    const real half_domain_length_z = 0.5*domain_length_z;

    // extract coordinates
    const real x = point[0];
    const real y = point[1];
    const real z = point[2];

    // return perturbed vertical velocity component
    // Reference: Eq.(2.30) -- P. Andersson, L. Brandt, A. Bottaro and D. S. Henningson, "On the breakdown of boundary layer streaks"
    // Reference for z_scale term: G. LODATO, P. CASTONGUAY AND A. JAMESON, "Discrete filter operators for large-eddy simulation using high-order spectral difference methods", Int. J. Numer. Meth. Fluids (2012)
    /*const */real F = C*exp(-pow((x-x_loc)/x_scale,2.0))*exp(-pow((y-y_loc)/y_scale,2.0))*cos(beta*(z+half_domain_length_z)/z_scale); // we do (z+half_domain_length_z) because reference has z\in[0,domain_length_z], whereas we center about the z-axis
    return F;
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_TurbulentChannelFlow<dim, nstate, real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    std::array<real,nstate> primitive_soln;

    //------------------------------------------------------
    // density
    //------------------------------------------------------
    // Reference: L. Wei, A. Pollard / Computers & Fluids 47 (2011) 85–100
    const real density = 1.0; // freestream non-dimensionalized
    primitive_soln[0] = density;
    
    //------------------------------------------------------
    // x-velocity
    //------------------------------------------------------
    // Reference: L. Wei, A. Pollard / Computers & Fluids 47 (2011) 85–100
    const real temperature = navier_stokes_physics.isothermal_wall_temperature;
    primitive_soln[1] = this->x_velocity(point,density,temperature);

    //------------------------------------------------------
    // y-velocity
    //------------------------------------------------------
    primitive_soln[2] = y_velocity(point);

    //------------------------------------------------------
    // z-velocity
    //------------------------------------------------------
    // Reference: R. Rossi / Journal of Computational Physics 228 (2009) 1639–1657
    primitive_soln[3] = 0.0;

    //------------------------------------------------------
    // pressure
    //------------------------------------------------------
    primitive_soln[4] = navier_stokes_physics.compute_pressure_from_density_temperature(density, temperature);

    //------------------------------------------------------
    // --> Get conservative solution
    //------------------------------------------------------
    std::array<real,nstate> conservative_soln = navier_stokes_physics.convert_primitive_to_conservative(primitive_soln);

    return conservative_soln[istate];
}

// ========================================================
// Turbulent Channel Flow -- Initial Condition (Turbulent x-velocity)
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_TurbulentChannelFlow_Turbulent<dim,nstate,real>
::InitialConditionFunction_TurbulentChannelFlow_Turbulent (
    const Physics::NavierStokes<dim,nstate,double> navier_stokes_physics_,
    const double channel_friction_velocity_reynolds_number_,
    const double domain_length_x_,
    const double domain_length_y_,
    const double domain_length_z_)
    : InitialConditionFunction_TurbulentChannelFlow<dim,nstate,real>(
        navier_stokes_physics_,
        channel_friction_velocity_reynolds_number_,
        domain_length_x_,
        domain_length_y_,
        domain_length_z_)
{}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_TurbulentChannelFlow_Turbulent<dim, nstate, real>
::x_velocity(const dealii::Point<dim,real> &point, const real density, const real temperature) const
{
    // Turbulent velocity profile using Reichart's law of the wall
    // -- apply initial condition symmetrically w.r.t. the top/bottom walls of the channel
    const real dist_from_wall = this->get_distance_from_wall(point);

    // Get the nondimensional (w.r.t. freestream) friction velocity
    const real viscosity_coefficient = this->navier_stokes_physics.compute_viscosity_coefficient_from_temperature(temperature);
    const real friction_velocity = viscosity_coefficient*this->channel_friction_velocity_reynolds_number/(density*this->half_channel_height*this->navier_stokes_physics.reynolds_number_inf);

    // Reichardt law of the wall (provides a smoothing between the linear and the log regions)
    // References: 
    /*  Frere, Carton de Wiart, Hillewaert, Chatelain, and Winckelmans 
        "Application of wall-models to discontinuous Galerkin LES", Phys. Fluids 29, 2017

        (Original paper) J. M.  Osterlund, A. V. Johansson, H. M. Nagib, and M. H. Hites, “A note
        on the overlap region in turbulent boundary layers,” Phys. Fluids 12, 1–4, (2000).
    */
    const real kappa = 0.38; // von Karman's constant
    const real C = 4.1;
    const real y_plus = this->navier_stokes_physics.reynolds_number_inf*density*friction_velocity*dist_from_wall/viscosity_coefficient;
    const real u_plus = (1.0/kappa)*log(1.0+kappa*y_plus) + (C - (1.0/kappa)*log(kappa))*(1.0 - exp(-y_plus/11.0) - (y_plus/11.0)*exp(-y_plus/3.0));
    const real x_velocity = u_plus*friction_velocity;
    return x_velocity;
}

// ========================================================
// Turbulent Channel Flow -- Initial Condition (Manufactured x-velocity)
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_TurbulentChannelFlow_Manufactured<dim,nstate,real>
::InitialConditionFunction_TurbulentChannelFlow_Manufactured (
    const Physics::NavierStokes<dim,nstate,double> navier_stokes_physics_,
    const double channel_friction_velocity_reynolds_number_,
    const double domain_length_x_,
    const double domain_length_y_,
    const double domain_length_z_)
    : InitialConditionFunction_TurbulentChannelFlow_Turbulent<dim,nstate,real>(
        navier_stokes_physics_,
        channel_friction_velocity_reynolds_number_,
        domain_length_x_,
        domain_length_y_,
        domain_length_z_)
{}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_TurbulentChannelFlow_Manufactured<dim, nstate, real>
::y_velocity(const dealii::Point<dim,real> &/*point*/) const
{
    // Manufactured velocity profile so that it is purely based on the x-velocity
    const real y_velocity = 0.0;
    return y_velocity;
}

// ========================================================
// NavierStokesBase -- Initial Condition
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_NavierStokesBase<dim,nstate,real>
::InitialConditionFunction_NavierStokesBase (
        Parameters::AllParameters const *const param)
    : InitialConditionFunction<dim,nstate,real>()
    , gamma_gas(param->euler_param.gamma_gas)
    , mach_inf(param->euler_param.mach_inf)
    , mach_inf_sqr(mach_inf*mach_inf)
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // Note that Euler primitive/conservative vars are the same as NS
    PHiLiP::Parameters::AllParameters parameters_euler = *param;
    parameters_euler.pde_type = Parameters::AllParameters::PartialDifferentialEquation::euler;
    this->euler_physics = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(
                Physics::PhysicsFactory<dim,dim+2,double>::create_Physics(&parameters_euler));
}

template <int dim, int nstate, typename real>
real InitialConditionFunction_NavierStokesBase<dim,nstate,real>
::convert_primitive_to_conversative_value(
    const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    
    std::array<real,nstate> soln_primitive;
    for (int i=0; i<nstate; ++i){
        soln_primitive[i] = primitive_value(point,i);
    }
    const std::array<real,nstate> soln_conservative = this->euler_physics->convert_primitive_to_conservative(soln_primitive);
    value = soln_conservative[istate];

    return value;
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_NavierStokesBase<dim, nstate, real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    value = convert_primitive_to_conversative_value(point,istate);
    return value;
}

// ========================================================
// TAYLOR GREEN VORTEX -- Initial Condition (Uniform density)
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_TaylorGreenVortex<dim,nstate,real>
::InitialConditionFunction_TaylorGreenVortex (
        Parameters::AllParameters const *const param)
    : InitialConditionFunction_NavierStokesBase<dim,nstate,real>(param)
{}

template <int dim, int nstate, typename real>
real InitialConditionFunction_TaylorGreenVortex<dim,nstate,real>
::primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    // Note: This is in non-dimensional form (free-stream values as reference)
    real value = 0.;
    if constexpr(dim == 3) {
        const real x = point[0], y = point[1], z = point[2];

        if(istate==0) {
            // density
            value = this->density(point);
        }
        if(istate==1) {
            // x-velocity
            value = sin(x)*cos(y)*cos(z);
        }
        if(istate==2) {
            // y-velocity
            value = -cos(x)*sin(y)*cos(z);
        }
        if(istate==3) {
            // z-velocity
            value = 0.0;
        }
        if(istate==4) {
            // pressure
            value = 1.0/(this->gamma_gas*this->mach_inf_sqr) + (1.0/16.0)*(cos(2.0*x)+cos(2.0*y))*(cos(2.0*z)+2.0);
        }
    }
    return value;
}

template <int dim, int nstate, typename real>
real InitialConditionFunction_TaylorGreenVortex<dim,nstate,real>
::density(const dealii::Point<dim,real> &/*point*/) const
{
    // Note: This is in non-dimensional form (free-stream values as reference)
    real value = 0.;
    // density
    value = 1.0;
    return value;
}

// ========================================================
// TAYLOR GREEN VORTEX -- Initial Condition (Isothermal density)
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_TaylorGreenVortex_Isothermal<dim,nstate,real>
::InitialConditionFunction_TaylorGreenVortex_Isothermal (
        Parameters::AllParameters const *const param)
    : InitialConditionFunction_TaylorGreenVortex<dim,nstate,real>(param)
{}

template <int dim, int nstate, typename real>
real InitialConditionFunction_TaylorGreenVortex_Isothermal<dim,nstate,real>
::density(const dealii::Point<dim,real> &point) const
{
    // Note: This is in non-dimensional form (free-stream values as reference)
    real value = 0.;
    // density
    value = this->primitive_value(point, 4); // get pressure
    value *= this->gamma_gas*this->mach_inf_sqr;
    return value;
}

// ========================================================
// Dipole Wall Collision -- Initial Condition
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_DipoleWallCollision<dim,nstate,real>
::InitialConditionFunction_DipoleWallCollision (
        Parameters::AllParameters const *const param,
        const real extremum_vorticity_value_,
        const real dipole_radius,
        const real dipole_axis_angle_wrt_x_axis_in_degrees)
    : InitialConditionFunction_NavierStokesBase<dim,nstate,real>(param)
    , extremum_vorticity_value(extremum_vorticity_value_)
    , r0(dipole_radius)
    , x1(dipole_radius*cos(dipole_axis_angle_wrt_x_axis_in_degrees*(3.141592653589793238/180.0)))
    , y1(dipole_radius*sin(dipole_axis_angle_wrt_x_axis_in_degrees*(3.141592653589793238/180.0)))
    , x2(-dipole_radius*cos(dipole_axis_angle_wrt_x_axis_in_degrees*(3.141592653589793238/180.0)))
    , y2(-dipole_radius*sin(dipole_axis_angle_wrt_x_axis_in_degrees*(3.141592653589793238/180.0)))
{ }

template <int dim, int nstate, typename real>
real InitialConditionFunction_DipoleWallCollision<dim,nstate,real>
::primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    // Note: This is in non-dimensional form (free-stream values as reference)
    real value = 0.;
    if constexpr(dim == 2) {
        const real x = point[0], y = point[1];
        // corresponding radii (non-dimensional)
        const real r1 = sqrt((x-this->x1)*(x-this->x1) + (y-this->y1)*(y-this->y1));
        const real r2 = sqrt((x-this->x2)*(x-this->x2) + (y-this->y2)*(y-this->y2));

        if(istate==0) {
            // density
            value = 1.0;
        }
        if(istate==1) {
            // x-velocity
            value = -0.5*abs(extremum_vorticity_value)*(y-this->y1)*exp(-(r1/this->r0)*(r1/this->r0))
                    +0.5*abs(extremum_vorticity_value)*(y-this->y2)*exp(-(r2/this->r0)*(r2/this->r0));
        }
        if(istate==2) {
            // y-velocity
            value = 0.5*abs(extremum_vorticity_value)*(x-this->x1)*exp(-(r1/this->r0)*(r1/this->r0))
                    -0.5*abs(extremum_vorticity_value)*(x-this->x2)*exp(-(r2/this->r0)*(r2/this->r0));
        }
        if(istate==3) {
            // pressure
            value = 1.0/(this->gamma_gas*this->mach_inf_sqr)
                    - (1.0/16.0)*pow(this->extremum_vorticity_value*this->r0,2.0)*(exp(-2.0*(r1/this->r0)*(r1/this->r0))+exp(-2.0*(r2/this->r0)*(r2/this->r0)));
        }
    }
    return value;
}

// ========================================================
// Dipole Wall Collision Normal -- Initial Condition
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_DipoleWallCollision_Normal<dim,nstate,real>
::InitialConditionFunction_DipoleWallCollision_Normal (
        Parameters::AllParameters const *const param)
    : InitialConditionFunction_DipoleWallCollision<dim,nstate,real>(
        param,
        299.528385375226, // reference: Keetels et al. 2007
        0.1, // dipole radius
        90.0) // dipole axis angle wrt to x-axis
{}

// ========================================================
// Dipole Wall Collision Oblique -- Initial Condition
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_DipoleWallCollision_Oblique<dim,nstate,real>
::InitialConditionFunction_DipoleWallCollision_Oblique (
        Parameters::AllParameters const *const param)
    : InitialConditionFunction_DipoleWallCollision<dim,nstate,real>(
        param,
        299.528385375226, // reference: Keetels et al. 2007
        0.1, // dipole radius
        30.0) // dipole axis angle wrt to x-axis
{}

// ========================================================
// 1D BURGERS REWIENSKI -- Initial Condition
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_BurgersRewienski<dim, nstate, real>
::InitialConditionFunction_BurgersRewienski ()
        : InitialConditionFunction<dim,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_BurgersRewienski<dim,nstate,real>
::value(const dealii::Point<dim,real> &/*point*/, const unsigned int /*istate*/) const
{
    real value = 1.0;
    return value;
}

// ========================================================
// 1D BURGERS VISCOUS -- Initial Condition
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_BurgersViscous<dim,nstate,real>
::InitialConditionFunction_BurgersViscous ()
        : InitialConditionFunction<dim,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_BurgersViscous<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    real value = 0;
    if(point[0] >= 0 && point[0] <= 0.25){
        value = sin(4*dealii::numbers::PI*point[0]);
    }
    return value;

}

// ========================================================
// 1D BURGERS Inviscid -- Initial Condition
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_BurgersInviscid<dim,nstate,real>
::InitialConditionFunction_BurgersInviscid ()
        : InitialConditionFunction<dim,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_BurgersInviscid<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    real value = 1.0;
    if constexpr(dim >= 1)
        value *= cos(dealii::numbers::PI*point[0]);
    if constexpr(dim >= 2)
        value *= cos(dealii::numbers::PI*point[1]);
    if constexpr(dim == 3)
        value *= cos(dealii::numbers::PI*point[2]);

    return value;
}

// ========================================================
// 1D BURGERS Inviscid Energy-- Initial Condition
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_BurgersInviscidEnergy<dim,nstate,real>
::InitialConditionFunction_BurgersInviscidEnergy ()
        : InitialConditionFunction<dim,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_BurgersInviscidEnergy<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    real value = 1.0;
    if constexpr(dim >= 1)
        value *= sin(dealii::numbers::PI*point[0]);
    if constexpr(dim >= 2)
        value *= sin(dealii::numbers::PI*point[1]);
    if constexpr(dim == 3)
        value *= sin(dealii::numbers::PI*point[2]);

    value += 0.01;
    return value;
}

// ========================================================
// Advection -- Initial Condition
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_AdvectionEnergy<dim,nstate,real>
::InitialConditionFunction_AdvectionEnergy ()
        : InitialConditionFunction<dim,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_AdvectionEnergy<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    real value = 1.0;
    if constexpr(dim >= 1)
        value *= exp(-20.0*point[0]*point[0]);
    if constexpr(dim >= 2)
        value *= exp(-20.0*point[1]*point[1]);
    if constexpr(dim == 3)
        value *= exp(-20.0*point[2]*point[2]);

    return value;
}

// ========================================================
// Advection OOA -- Initial Condition
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_Advection<dim,nstate,real>
::InitialConditionFunction_Advection()
        : InitialConditionFunction<dim,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_Advection<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    real value = 1.0;
    if constexpr(dim >= 1)
        value *= sin(2.0*dealii::numbers::PI*point[0]);
    if constexpr(dim >= 2)
        value *= sin(2.0*dealii::numbers::PI*point[1]);
    if constexpr(dim == 3)
        value *= sin(2.0*dealii::numbers::PI*point[2]);

    return value;
}

// ========================================================
// Convection_diffusion -- Initial Condition
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_ConvDiff<dim,nstate,real>
::InitialConditionFunction_ConvDiff ()
        : InitialConditionFunction<dim,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_ConvDiff<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    real value = 1.0;
    if constexpr(dim >= 1)
        value *= sin(dealii::numbers::PI*point[0]);
    if constexpr(dim >= 2)
        value *= sin(dealii::numbers::PI*point[1]);
    if constexpr(dim == 3)
        value *= sin(dealii::numbers::PI*point[2]);

    return value;
}

// ========================================================
// Convection_diffusion Energy -- Initial Condition
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_ConvDiffEnergy<dim,nstate,real>
::InitialConditionFunction_ConvDiffEnergy ()
        : InitialConditionFunction<dim,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_ConvDiffEnergy<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    real value = 1.0;
    if constexpr(dim >= 1)
        value *= sin(dealii::numbers::PI*point[0]);
    if constexpr(dim >= 2)
        value *= sin(dealii::numbers::PI*point[1]);
    if constexpr(dim == 3)
        value *= sin(dealii::numbers::PI*point[2]);

    value += 0.1;

    return value;
}

// ========================================================
// 1D SINE -- Initial Condition for advection_explicit_time_study
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_1DSine<dim,nstate,real>
::InitialConditionFunction_1DSine ()
        : InitialConditionFunction<dim,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_1DSine<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    real value = 0;
    real pi = dealii::numbers::PI;
    if(point[0] >= 0.0 && point[0] <= 2.0){
        value = sin(2*pi*point[0]/2.0);
    }
    return value;
}

// ========================================================
// Inviscid Isentropic Vortex
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_IsentropicVortex<dim,nstate,real>
::InitialConditionFunction_IsentropicVortex(
        Parameters::AllParameters const *const param)
        : InitialConditionFunction<dim,nstate,real>()
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // This test should only be used for Euler
    this->euler_physics = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(
                Physics::PhysicsFactory<dim,dim+2,double>::create_Physics(param));
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_IsentropicVortex<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    // Setting constants
    const double pi = dealii::numbers::PI;
    const double gam = 1.4;
    const double M_infty = sqrt(2/gam);
    const double R = 1;
    const double sigma = 1;
    const double beta = M_infty * 5 * sqrt(2.0)/4.0/pi * exp(1.0/2.0);
    const double alpha = pi/4; //rad

    // Centre of the vortex  at t=0
    const double x0 = 0.0;
    const double y0 = 0.0;
    const double x = point[0] - x0;
    const double y = point[1] - y0;

    const double Omega = beta * exp(-0.5/sigma/sigma* (x/R * x/R + y/R * y/R));
    const double delta_Ux = -y/R * Omega;
    const double delta_Uy =  x/R * Omega;
    const double delta_T  = -(gam-1.0)/2.0 * Omega * Omega;

    // Primitive
    std::array<real,nstate> soln_primitive;
    soln_primitive[0] = pow((1 + delta_T), 1.0/(gam-1.0));
    soln_primitive[1] = M_infty * cos(alpha) + delta_Ux;
    soln_primitive[2] = M_infty * sin(alpha) + delta_Uy;
    #if PHILIP_DIM==3
    soln_primitive[3] = 0;
    #endif
    soln_primitive[nstate-1] = 1.0/gam*pow(1+delta_T, gam/(gam-1.0));

    const std::array<real,nstate> soln_conservative = this->euler_physics->convert_primitive_to_conservative(soln_primitive);
    return soln_conservative[istate];
}

// ========================================================
// KELVIN-HELMHOLTZ INSTABILITY
// See Chan et al., On the entropy projection..., 2022, Pg. 15
//     Note that some equations are not typed correctly
//     See github.com/trixi-framework/paper-2022-robustness-entropy-projection
//     for initial condition which is implemented herein
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_KHI<dim,nstate,real>
::InitialConditionFunction_KHI (
        Parameters::AllParameters const *const param)
    : InitialConditionFunction<dim,nstate,real>()
    , atwood_number(param->flow_solver_param.atwood_number)
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // This test should only be used for Euler
    this->euler_physics = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(
                Physics::PhysicsFactory<dim,dim+2,double>::create_Physics(param));
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_KHI<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    const double pi = dealii::numbers::PI;
    
    const double B = 0.5 * (tanh(15*point[1] + 7.5) - tanh(15*point[1] - 7.5));

    const double rho1 = 0.5;
    const double rho2 = rho1 * (1 + atwood_number) / (1 - atwood_number);

    std::array<real,nstate> soln_primitive;
    soln_primitive[0] = rho1 + B * (rho2-rho1);
    soln_primitive[nstate-1] = 1;
    soln_primitive[1] = B - 0.5;
    soln_primitive[2] = 0.1 * sin(2 * pi * point[0]);

    const std::array<real,nstate> soln_conservative = this->euler_physics->convert_primitive_to_conservative(soln_primitive);
    return soln_conservative[istate];
}

// ========================================================
// 1D Sod Shock tube -- Initial Condition
// See Chen & Shu, Entropy stable high order..., 2017, Pg. 25
// 2D and 3D can be run by extruding grid in those directions
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_SodShockTube<dim,nstate,real>
::InitialConditionFunction_SodShockTube (
        Parameters::AllParameters const* const param)
        : InitialConditionFunction_NavierStokesBase<dim,nstate,real>(param)
{}

template <int dim, int nstate, typename real>
real InitialConditionFunction_SodShockTube<dim, nstate, real>
::primitive_value(const dealii::Point<dim, real>& point, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr (nstate == (dim+2)) {
        const real x = point[0];
        if (x < 0) {
            if (istate == 0) {
                // density
                value = 1.0;
            }
            if (istate == nstate - 1) {
                // pressure
                value = 1.0;
            }
        } else {
            if (istate == 0) {
                // density
                value = 0.125;
            }
            if (istate == nstate - 1) {
                // pressure
                value = 0.1;
            }
        }
    }

    return value;
}

// ========================================================
// 2D/3D Explosion Problem (Cylindrical and Spherical extension of Sod Shock Tube) -- Initial Condition
// See Dumbser & Loubere, A simple robust and accurate a posteriori sub-cell..., 2016, Pg. 17
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_ExplosionProblem<dim, nstate, real>
::InitialConditionFunction_ExplosionProblem(
    Parameters::AllParameters const* const param)
    : InitialConditionFunction_NavierStokesBase<dim,nstate,real>(param)
{}

template <int dim, int nstate, typename real>
real InitialConditionFunction_ExplosionProblem<dim, nstate, real>
::primitive_value(const dealii::Point<dim, real>& point, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr (dim == 2 && nstate == (dim + 2)) {
        const real x = point[0];
        const real y = point[1];
        if (sqrt(pow(x,2)+ pow(y,2)) <= 0.5){
            if (istate == 0) {
                // density
                value = 1.0;
            }
            if (istate == 1) {
                // x-velocity
                value = 0.0;
            }
            if (istate == 2) {
                // x-velocity
                value = 0.0;
            }
            if (istate == 3) {
                // pressure
                value = 1.0;
            }
        }
        else {
            if (istate == 0) {
                // density
                value = 0.125;
            }
            if (istate == 1) {
                // x-velocity
                value = 0.0;
            }
            if (istate == 2) {
                // x-velocity
                value = 0.0;
            }
            if (istate == 3) {
                // pressure
                value = 0.1;
            }
        }
    }

    if constexpr (dim == 3 && nstate == (dim + 2)) {
        const real x = point[0];
        const real y = point[1];
        const real z = point[2];
        if (sqrt(pow(x,2) + pow(y,2) + pow(z,2)) <= 0.5){
            if (istate == 0) {
                // density
                value = 1.0;
            }
            if (istate == 1) {
                // x-velocity
                value = 0.0;
            }
            if (istate == 2) {
                // y-velocity
                value = 0.0;
            }
            if (istate == 3) {
                // z-velocity
                value = 0.0;
            }
            if (istate == 4) {
                // pressure
                value = 1.0;
            }
        }
        else {
            if (istate == 0) {
                // density
                value = 0.125;
            }
            if (istate == 1) {
                // x-velocity
                value = 0.0;
            }
            if (istate == 2) {
                // y-velocity
                value = 0.0;
            }
            if (istate == 3) {
                // z-velocity
                value = 0.0;
            }
            if (istate == 4) {
                // pressure
                value = 0.1;
            }
        }
    }

    return value;
}

// ========================================================
// 1D Leblanc Shock tube -- Initial Condition
// 2D Cylindrical Extension of Leblanc -- Initial Condition
// See Zhang & Shu, On positivity-preserving..., 2010 Pg. 14 for 1D
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_LeblancShockTube<dim,nstate,real>
::InitialConditionFunction_LeblancShockTube(
    Parameters::AllParameters const* const param)
    : InitialConditionFunction_NavierStokesBase<dim,nstate,real>(param)
{}

template <int dim, int nstate, typename real>
real InitialConditionFunction_LeblancShockTube<dim, nstate, real>
::primitive_value(const dealii::Point<dim, real>& point, const unsigned int istate) const
{
    real value = 0.0;
    if (dim == 1 && nstate == (dim + 2)) {
        const real x = point[0];
        if (x < 0) {
            if (istate == 0) {
                // density
                value = 2.0;
            }
            if (istate == 1) {
                // x-velocity
                value = 0.0;
            }
            if (istate == 2) {
                // pressure
                value = pow(10.0, 9.0);
            }
        }
        else {
            if (istate == 0) {
                // density
                value = 0.001;
            }
            if (istate == 1) {
                // x-velocity
                value = 0.0;
            }
            if (istate == 2) {
                // pressure
                value = 1.0;
            }
        }
    }

    if constexpr (dim == 2 && nstate == (dim + 2)) {
        const real x = point[0];
        const real y = point[1];
        if (sqrt(pow(x,2)+ pow(y,2)) <= 10.0){
            if (istate == 0) {
                // density
                value = 2.0;
            }
            if (istate == 1) {
                // x-velocity
                value = 0.0;
            }
            if (istate == 2) {
                // x-velocity
                value = 0.0;
            }
            if (istate == 3) {
                // pressure
                value = pow(10.0, 9.0);
            }
        }
        else {
            if (istate == 0) {
                // density
                value = 0.001;
            }
            if (istate == 1) {
                // x-velocity
                value = 0.0;
            }
            if (istate == 2) {
                // x-velocity
                value = 0.0;
            }
            if (istate == 3) {
                // pressure
                value = 1.0;
            }
        }
    }
    return value;
}

// ========================================================
// 1D Shu-Osher Problem -- Initial Condition
// See Johnsen et al., Assessment of high-resolution..., 2010 Pg. 7
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_ShuOsherProblem<dim, nstate, real>
::InitialConditionFunction_ShuOsherProblem(
    Parameters::AllParameters const* const param)
    : InitialConditionFunction_NavierStokesBase<dim,nstate,real>(param)
{}

template <int dim, int nstate, typename real>
real InitialConditionFunction_ShuOsherProblem<dim, nstate, real>
::primitive_value(const dealii::Point<dim, real>& point, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr (dim == 1 && nstate == (dim + 2)) {
        const real x = point[0];
        if (x < -4.0) {
            if (istate == 0) {
                // density
                value = 3.857143;
            }
            else if (istate == 1) {
                // x-velocity
                value = 2.629369;
            }
            else if (istate == 2) {
                // pressure
                value = 10.33333;
            }
        }
        else {
            if (istate == 0) {
                // density
                value = 1 + 0.2 * sin(5 * x);
            }
            else if (istate == 1) {
                // x-velocity
                value = 0.0;
            }
            else if (istate == 2) {
                // pressure
                value = 1.0;
            }
        }
    }
    return value;
}

// ========================================================
// 2D Low Density Euler -- Initial Condition
// See Zhang & Shu, On positivity-preserving..., 2010 Pg. 10
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_LowDensity2D<dim,nstate,real>
::InitialConditionFunction_LowDensity2D(
    Parameters::AllParameters const* const param)
    : InitialConditionFunction_NavierStokesBase<dim, nstate, real>(param)
{}

template <int dim, int nstate, typename real>
real InitialConditionFunction_LowDensity2D<dim, nstate, real>
::primitive_value(const dealii::Point<dim, real>& point, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr (dim == 2 && nstate == (dim + 2)) {
        const real x = point[0];
        const real y = point[1];
        if (istate == 0) {
            // density
            value = 1 + 0.99 * sin(x + y);
        }
        if (istate == 1) {
            // x-velocity
            value = 1.0;
        }
        if (istate == 2) {
            // y-velocity
            value = 1.0;
        }
        if (istate == 3) {
            // pressure
            value = 1.0;
        }
    }
    return value;
}

// ========================================================
// Double Mach Reflection Problem (2D) -- Initial Condition
// INCLUDE REFERENCE LATER
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_DoubleMachReflection<dim, nstate, real>
::InitialConditionFunction_DoubleMachReflection(
    Parameters::AllParameters const* const param)
    : InitialConditionFunction_NavierStokesBase<dim, nstate, real>(param)
{}

template <int dim, int nstate, typename real>
real InitialConditionFunction_DoubleMachReflection<dim, nstate, real>
::primitive_value(const dealii::Point<dim, real>& point, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr (dim == 2 && nstate == (dim + 2)) {
        const real x = point[0];
        const real y = point[1];
        if (y > sqrt(3)*(x - (1.0/6.0))) {
            if (istate == 0) {
                // density
                value = 8.0;
            }
            else if (istate == 1) {
                // x-velocity
                value = 33.0*sqrt(3.0)/8.0;
            }
            else if (istate == 2) {
                // y-velocity
                value = -33.0/8.0;
            }
            else if (istate == 3) {
                // pressure
                value = 116.5;
            }
        }
        else {
            if (istate == 0) {
                // density
                value = 1.4;
            }
            else if (istate == 1) {
                // x-velocity
                value = 0.0;
            }
            else if (istate == 2) {
                // y-velocity
                value = 0.0;
            }
            else if (istate == 3) {
                // pressure
                value = 1.0;
            }
        }
    }
    return value;
}

// ========================================================
// Sedov Blast Wave (2D) -- Initial Condition
// INCLUDE REFERENCE LATER
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_SedovBlastWave<dim, nstate, real>
::InitialConditionFunction_SedovBlastWave(
    Parameters::AllParameters const* const param)
    : InitialConditionFunction_NavierStokesBase<dim, nstate, real>(param)
    , gamma_gas(param->euler_param.gamma_gas)
    , n_subdivisions(param->flow_solver_param.number_of_grid_elements_x)
    , xmax(param->flow_solver_param.grid_xmax)
    , xmin(param->flow_solver_param.grid_xmin)
{
    this->h = (xmax-xmin)/double(n_subdivisions);
}


template <int dim, int nstate, typename real>
real InitialConditionFunction_SedovBlastWave<dim, nstate, real>
::primitive_value(const dealii::Point<dim, real>& point, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr (dim == 2 && nstate == (dim + 2)) {
        const real x = point[0];
        const real y = point[1];
        real r = sqrt(pow(x,2)+pow(y,2));
        //real r_0 = 4*this->h;

        if (istate == 0) {
            // density
            value = 1.0;
        }
        else if (istate == 1) {
            // x-velocity
            value = 0.0;
        }
        else if (istate == 2) {
            // y-velocity
            value = 0.0;
        }
        else if (istate == 3) {
            // pressure
            if(r < this->h)//r_0)
                value = (this->gamma_gas - 1.0)*(0.244816/pow(this->h,2));///(this->gamma_gas - 1.0)/(dealii::numbers::PI*pow(r_0,2));
            else
                value = 4e-13;//1e-5;
        }

    }
    return value;
}

// ========================================================
// Mach 3 Wind Tunnel with a Step Problem (2D) -- Initial Condition
// INCLUDE REFERENCE LATER
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_Mach3WindTunnel<dim, nstate, real>
::InitialConditionFunction_Mach3WindTunnel(
    Parameters::AllParameters const* const param)
    : InitialConditionFunction_NavierStokesBase<dim, nstate, real>(param)
{}

template <int dim, int nstate, typename real>
real InitialConditionFunction_Mach3WindTunnel<dim, nstate, real>
::primitive_value(const dealii::Point<dim, real>& /*point*/, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr (dim == 2 && nstate == (dim + 2)) {
        if (istate == 0) {
            // density
            value = 1.4;
        }
        else if (istate == 1) {
            // x-velocity
            value = 3.0;
        }
        else if (istate == 2) {
            // y-velocity
            value = 0.0;
        }
        else if (istate == 3) {
            // pressure
            value = 1.0;
        }
    }
    return value;
}

// ========================================================
// Shock Diffraction (backwards facing step) (2D) -- Initial Condition
// INCLUDE REFERENCE LATER
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_ShockDiffraction<dim, nstate, real>
::InitialConditionFunction_ShockDiffraction(
    Parameters::AllParameters const* const param)
    : InitialConditionFunction_NavierStokesBase<dim, nstate, real>(param)
{}

template <int dim, int nstate, typename real>
real InitialConditionFunction_ShockDiffraction<dim, nstate, real>
::primitive_value(const dealii::Point<dim, real>& point, const unsigned int istate) const
{
    real value = 0.0;
    real x = point[0];
    //real y = point[1];
    if constexpr (dim == 2 && nstate == (dim + 2)) {
        if (x <= 0.5) {
            if (istate == 0) {
                // density
                value = 7.041132906907898;
            }
            else if (istate == 1) {
                // x-velocity
                value = 4.07794695481336;
            }
            else if (istate == 2) {
                // y-velocity
                value = 0.0;
            }
            else if (istate == 3) {
                // pressure
                value = 30.05945;
            }
        }
        else {
           if (istate == 0) {
               // density
               value = 1.4;
           }
           else if (istate == 1) {
               // x-velocity
               value = 0.0;
           }
           else if (istate == 2) {
               // y-velocity
               value = 0.0;
           }
           else if (istate == 3) {
               // pressure
               value = 1.0;
           }
        }
    }
    return value;
}


// ========================================================
// Astrophysical Mach Jet (2D) -- Initial Condition
// INCLUDE REFERENCE LATER
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_AstrophysicalJet<dim, nstate, real>
::InitialConditionFunction_AstrophysicalJet(
    Parameters::AllParameters const* const param)
    : InitialConditionFunction_NavierStokesBase<dim, nstate, real>(param)
{}

template <int dim, int nstate, typename real>
real InitialConditionFunction_AstrophysicalJet<dim, nstate, real>
::primitive_value(const dealii::Point<dim, real>& /*point*/, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr (dim == 2 && nstate == (dim + 2)) {

        if (istate == 0) {
            // density
            value = 0.5;
        }
        else if (istate == 1) {
            // x-velocity
            value = 0.0;
        }
        else if (istate == 2) {
            // y-velocity
            value = 0.0;
        }
        else if (istate == 3) {
            // pressure
            value = 0.4127;
        }
    }
    return value;
}

// ========================================================
// Viscous Shock Tube -- Initial Condition
/** Reference: Guermond, Jean-Luc, et al. "Second-order invariant 
    domain preserving approximation of the compressible Navier–Stokes equations."
    Computer Methods in Applied Mechanics and Engineering 375 (2021): 113608.
    */
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_ViscousShockTube<dim, nstate, real>
::InitialConditionFunction_ViscousShockTube(
    Parameters::AllParameters const* const param)
    : InitialConditionFunction_NavierStokesBase<dim, nstate, real>(param)
    , rho_0(param->flow_solver_param.vst_rho_0)
    , v_0(param->flow_solver_param.vst_v_0)
    , v_inf(param->flow_solver_param.vst_v_inf)
{}
template <int dim, int nstate, typename real>
real InitialConditionFunction_ViscousShockTube<dim, nstate, real>
::primitive_value(const dealii::Point<dim, real>& point, const unsigned int istate) const
{
    real value = 0.0;
    real x = point[0];
    // real m_0 = rho_0*v_0;
    real v_rhJump = (this->gamma_gas-1.0 + (2.0/this->mach_inf_sqr))/(this->gamma_gas + 1.0);
    // real v_01 = sqrt(v_0*v_1);
    if constexpr (dim == 1 && nstate == (dim + 2)) {
        if(x < 0.0) {
            if (istate == 0) {
                // density
                value = rho_0;
            }
            else if (istate == 1) {
                // x-velocity
                value = v_0 + v_inf;
            }
            else if (istate == 2) {
                // pressure
                value = rho_0*(1.0/this->gamma_gas)*pow(v_0,2.0)*(1.0/this->mach_inf_sqr);
            }
        }
        else {
            if (istate == 0) {
                // density
                value = pow(v_rhJump,-1.0);
            }
            else if (istate == 1) {
                // x-velocity
                value = v_rhJump + v_inf;
            }
            else if (istate == 2) {
                // pressure
                real coeff = (1.0 + (2.0/((this->gamma_gas - 1)*this->mach_inf_sqr)) - pow(v_rhJump, 2.0));
                value = (this->gamma_gas-1.0)*pow(v_rhJump,-1.0)*(coeff/(2.0*this->gamma_gas));
            }
        }
    }
    return value;
}

// ========================================================
// 2D Daru-Tenaud Problem -- Initial Condition
/** Reference: Daru, Virginie, and Christian Tenaud. "Numerical simulation of the viscous
     shock tube problem by using a high resolution monotonicity-preserving scheme." 
     Computers & Fluids 38.3 (2009): 664-676.
*/
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_DaruTenaudShockTube<dim, nstate, real>
::InitialConditionFunction_DaruTenaudShockTube(
    Parameters::AllParameters const* const param)
    : InitialConditionFunction_NavierStokesBase<dim, nstate, real>(param)
{}

template <int dim, int nstate, typename real>
real InitialConditionFunction_DaruTenaudShockTube<dim, nstate, real>
::primitive_value(const dealii::Point<dim, real>& point, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr (dim == 2 && nstate == (dim + 2)) {
        const real x = point[0];
        if (x < 0.5) {
            if (istate == 0) {
                // density
                value = 120;
            }
            else if (istate == 1) {
                // x-velocity
                value = 0;
            }
            else if (istate == 2) {
                // y-velocity
                value = 0;
            }
            else if (istate == 3) {
                value = 120.0/this->gamma_gas;
            }
        }
        else {
            if (istate == 0) {
                // density
                value = 1.2;
            }
            else if (istate == 1) {
                // x-velocity
                value = 0;
            }
            else if (istate == 2) {
                // y-velocity
                value = 0;
            }
            else if (istate == 3) {
                value = 1.2/this->gamma_gas;
            }
        }
    }
    return value;
}

// ========================================================
// ZERO INITIAL CONDITION
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_Zero<dim,nstate,real>
::InitialConditionFunction_Zero()
    : InitialConditionFunction<dim,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nstate, typename real>
real InitialConditionFunction_Zero<dim, nstate, real>
::value(const dealii::Point<dim,real> &/*point*/, const unsigned int /*istate*/) const
{
    return 0.0;
}

// =========================================================
// Initial Condition Factory
// =========================================================
template <int dim, int nstate, typename real>
std::shared_ptr<InitialConditionFunction<dim, nstate, real>>
InitialConditionFactory<dim,nstate, real>::create_InitialConditionFunction(
    Parameters::AllParameters const *const param)
{
    // Get the flow case type
    const FlowCaseEnum flow_type = param->flow_solver_param.flow_case_type;
    if (flow_type == FlowCaseEnum::taylor_green_vortex) {
        if constexpr (dim==3 && nstate==dim+2){ 
            // Get the density initial condition type
            const DensityInitialConditionEnum density_initial_condition_type = param->flow_solver_param.density_initial_condition_type;
            if(density_initial_condition_type == DensityInitialConditionEnum::uniform) {
                return std::make_shared<InitialConditionFunction_TaylorGreenVortex<dim,nstate,real> >(
                        param);
            } else if (density_initial_condition_type == DensityInitialConditionEnum::isothermal) {
                return std::make_shared<InitialConditionFunction_TaylorGreenVortex_Isothermal<dim,nstate,real> >(
                        param);
            }
        }
    } else if (flow_type == FlowCaseEnum::dipole_wall_collision_normal) {
        if constexpr (dim==2 && nstate==dim+2){ 
            return std::make_shared<InitialConditionFunction_DipoleWallCollision_Normal<dim,nstate,real> >(
                        param);
        }
    } else if (flow_type == FlowCaseEnum::dipole_wall_collision_oblique) {
        if constexpr (dim==2 && nstate==dim+2){ 
            return std::make_shared<InitialConditionFunction_DipoleWallCollision_Oblique<dim,nstate,real> >(
                        param);
        }
    } else if (flow_type == FlowCaseEnum::decaying_homogeneous_isotropic_turbulence) {
        if constexpr (dim==3 && nstate==dim+2) return nullptr; // nullptr since DHIT case initializes values from file
    } else if (flow_type == FlowCaseEnum::burgers_rewienski_snapshot) {
        if constexpr (dim==1 && nstate==1) return std::make_shared<InitialConditionFunction_BurgersRewienski<dim,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::burgers_viscous_snapshot) {
        if constexpr (dim==1 && nstate==1) return std::make_shared<InitialConditionFunction_BurgersViscous<dim,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::naca0012 || flow_type == FlowCaseEnum::gaussian_bump) {
        if constexpr ((dim==2 || dim==3) && nstate==dim+2) {
            Physics::Euler<dim,nstate,double> euler_physics_double = Physics::Euler<dim, nstate, double>(
                    param,
                    param->euler_param.ref_length,
                    param->euler_param.gamma_gas,
                    param->euler_param.mach_inf,
                    param->euler_param.angle_of_attack,
                    param->euler_param.side_slip_angle);
            return std::make_shared<FreeStreamInitialConditions<dim,nstate,real>>(euler_physics_double);
        }
    } else if (flow_type == FlowCaseEnum::burgers_inviscid && param->use_energy==false) {
        if constexpr (nstate==dim && dim<3) return std::make_shared<InitialConditionFunction_BurgersInviscid<dim, nstate, real> >();
    } else if (flow_type == FlowCaseEnum::burgers_inviscid && param->use_energy==true) {
        if constexpr (dim==1 && nstate==1) return std::make_shared<InitialConditionFunction_BurgersInviscidEnergy<dim,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::advection && param->use_energy==true) {
        if constexpr (nstate==1) return std::make_shared<InitialConditionFunction_AdvectionEnergy<dim,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::advection && param->use_energy==false) {
        if constexpr (nstate==1) return std::make_shared<InitialConditionFunction_Advection<dim,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::convection_diffusion && !param->use_energy) {
        if constexpr (nstate==1) return std::make_shared<InitialConditionFunction_ConvDiff<dim,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::convection_diffusion && param->use_energy) {
        return std::make_shared<InitialConditionFunction_ConvDiffEnergy<dim,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::periodic_1D_unsteady) {
        if constexpr (dim==1 && nstate==1) return std::make_shared<InitialConditionFunction_1DSine<dim,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::isentropic_vortex) {
        if constexpr (dim>1 && nstate==dim+2) return std::make_shared<InitialConditionFunction_IsentropicVortex<dim,nstate,real> > (param);
    } else if (flow_type == FlowCaseEnum::kelvin_helmholtz_instability) {
        if constexpr (dim>1 && nstate==dim+2) return std::make_shared<InitialConditionFunction_KHI<dim,nstate,real> > (param);
    } else if (flow_type == FlowCaseEnum::non_periodic_cube_flow) {
        if constexpr (dim==2 && nstate==1)  return std::make_shared<InitialConditionFunction_Zero<dim,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::channel_flow) {
        if constexpr (dim==3 && nstate==dim+2) {
            Physics::NavierStokes<dim,nstate,double> navier_stokes_physics_double = Physics::NavierStokes<dim, nstate, double>(
                    param,
                    param->euler_param.ref_length,
                    param->euler_param.gamma_gas,
                    param->euler_param.mach_inf,
                    param->euler_param.angle_of_attack,
                    param->euler_param.side_slip_angle,
                    param->navier_stokes_param.prandtl_number,
                    param->navier_stokes_param.reynolds_number_inf,
                    param->navier_stokes_param.use_constant_viscosity,
                    param->navier_stokes_param.nondimensionalized_constant_viscosity,
                    param->navier_stokes_param.temperature_inf,
                    param->navier_stokes_param.nondimensionalized_isothermal_wall_temperature,
                    param->navier_stokes_param.thermal_boundary_condition_type,
                    nullptr,
                    param->two_point_num_flux_type);
            // Get the x-velocity initial condition type
            const XVelocityInitialConditionEnum xvelocity_initial_condition_type = param->flow_solver_param.xvelocity_initial_condition_type;
            if(xvelocity_initial_condition_type == XVelocityInitialConditionEnum::laminar) {
                return std::make_shared<InitialConditionFunction_TurbulentChannelFlow<dim,nstate,real>>(
                    navier_stokes_physics_double,
                    param->flow_solver_param.turbulent_channel_friction_velocity_reynolds_number,
                    param->flow_solver_param.turbulent_channel_domain_length_x_direction,
                    param->flow_solver_param.turbulent_channel_domain_length_y_direction,
                    param->flow_solver_param.turbulent_channel_domain_length_z_direction);
            } else if(xvelocity_initial_condition_type == XVelocityInitialConditionEnum::turbulent) {
                return std::make_shared<InitialConditionFunction_TurbulentChannelFlow_Turbulent<dim,nstate,real>>(
                    navier_stokes_physics_double,
                    param->flow_solver_param.turbulent_channel_friction_velocity_reynolds_number,
                    param->flow_solver_param.turbulent_channel_domain_length_x_direction,
                    param->flow_solver_param.turbulent_channel_domain_length_y_direction,
                    param->flow_solver_param.turbulent_channel_domain_length_z_direction);
            } else if(xvelocity_initial_condition_type == XVelocityInitialConditionEnum::manufactured) {
                return std::make_shared<InitialConditionFunction_TurbulentChannelFlow_Manufactured<dim,nstate,real>>(
                    navier_stokes_physics_double,
                    param->flow_solver_param.turbulent_channel_friction_velocity_reynolds_number,
                    param->flow_solver_param.turbulent_channel_domain_length_x_direction,
                    param->flow_solver_param.turbulent_channel_domain_length_y_direction,
                    param->flow_solver_param.turbulent_channel_domain_length_z_direction);
            }
        }
    } else if (flow_type == FlowCaseEnum::sod_shock_tube) {
        if constexpr (dim >= 1 && nstate == dim+2)  return std::make_shared<InitialConditionFunction_SodShockTube<dim,nstate,real> > (param);
    } else if (flow_type == FlowCaseEnum::explosion_problem) {
        if constexpr (dim>1 && nstate==dim+2)  return std::make_shared<InitialConditionFunction_ExplosionProblem<dim,nstate,real> > (param);
    } else if (flow_type == FlowCaseEnum::low_density_2d) {
        if constexpr (dim == 2 && nstate == dim+2)  return std::make_shared<InitialConditionFunction_LowDensity2D<dim,nstate,real> > (param);
    } else if (flow_type == FlowCaseEnum::leblanc_shock_tube) {
        if constexpr (dim < 3 && nstate == dim+2)  return std::make_shared<InitialConditionFunction_LeblancShockTube<dim,nstate,real> > (param);
    } else if (flow_type == FlowCaseEnum::shu_osher_problem) {
        if constexpr (dim == 1 && nstate == dim + 2)  return std::make_shared<InitialConditionFunction_ShuOsherProblem<dim, nstate, real> >(param);
    } else if (flow_type == FlowCaseEnum::double_mach_reflection) {
        if constexpr (dim == 2 && nstate == dim + 2)  return std::make_shared<InitialConditionFunction_DoubleMachReflection<dim, nstate, real> >(param);
    } else if (flow_type == FlowCaseEnum::mach_3_wind_tunnel) {
        if constexpr (dim == 2 && nstate == dim + 2)  return std::make_shared<InitialConditionFunction_Mach3WindTunnel<dim, nstate, real> >(param);
    } else if (flow_type == FlowCaseEnum::sedov_blast_wave) {
        if constexpr (dim == 2 && nstate == dim + 2)  return std::make_shared<InitialConditionFunction_SedovBlastWave<dim, nstate, real> >(param);
    } else if (flow_type == FlowCaseEnum::shock_diffraction) {
        if constexpr (dim == 2 && nstate == dim + 2)  return std::make_shared<InitialConditionFunction_ShockDiffraction<dim, nstate, real> >(param);
    } else if (flow_type == FlowCaseEnum::astrophysical_jet) {
        if constexpr (dim == 2 && nstate == dim + 2)  return std::make_shared<InitialConditionFunction_AstrophysicalJet<dim, nstate, real> >(param);
    } else if (flow_type == FlowCaseEnum::viscous_shock_tube) {
        if constexpr (dim == 1 && nstate == dim + 2)  return std::make_shared<InitialConditionFunction_ViscousShockTube<dim, nstate, real> >(param);
    } else if (flow_type == FlowCaseEnum::daru_tenaud) {
        if constexpr (dim == 2 && nstate == dim + 2)  return std::make_shared<InitialConditionFunction_DaruTenaudShockTube<dim, nstate, real> >(param);
    } else if (flow_type == FlowCaseEnum::advection_limiter) {
        if constexpr (dim < 3 && nstate == 1)  return std::make_shared<InitialConditionFunction_Advection<dim, nstate, real> >();
    } else if (flow_type == FlowCaseEnum::burgers_limiter) {
        if constexpr (nstate==dim && dim<3) return std::make_shared<InitialConditionFunction_BurgersInviscid<dim, nstate, real> >();
    } else {
        std::cout << "Invalid Flow Case Type. You probably forgot to add it to the list of flow cases in initial_condition_function.cpp" << std::endl;
        std::abort();
    }
    return nullptr;
}

template class InitialConditionFunction <PHILIP_DIM, 1, double>;
template class InitialConditionFunction <PHILIP_DIM, 2, double>;
template class InitialConditionFunction <PHILIP_DIM, 3, double>;
template class InitialConditionFunction <PHILIP_DIM, 4, double>;
template class InitialConditionFunction <PHILIP_DIM, 5, double>;
template class InitialConditionFunction <PHILIP_DIM, 6, double>;
template class InitialConditionFactory <PHILIP_DIM, 1, double>;
template class InitialConditionFactory <PHILIP_DIM, 2, double>;
template class InitialConditionFactory <PHILIP_DIM, 3, double>;
template class InitialConditionFactory <PHILIP_DIM, 4, double>;
template class InitialConditionFactory <PHILIP_DIM, 5, double>;
template class InitialConditionFactory <PHILIP_DIM, 6, double>;
#if PHILIP_DIM==1
template class InitialConditionFunction_BurgersViscous <PHILIP_DIM, 1, double>;
template class InitialConditionFunction_BurgersRewienski <PHILIP_DIM, 1, double>;
template class InitialConditionFunction_BurgersInviscidEnergy <PHILIP_DIM, 1, double>;
template class InitialConditionFunction_ShuOsherProblem <PHILIP_DIM, PHILIP_DIM + 2, double>;
template class InitialConditionFunction_ViscousShockTube<PHILIP_DIM, PHILIP_DIM + 2, double>;
#endif

#if PHILIP_DIM==3
template class InitialConditionFunction_TaylorGreenVortex <PHILIP_DIM, PHILIP_DIM+2, double>;
template class InitialConditionFunction_TaylorGreenVortex_Isothermal <PHILIP_DIM, PHILIP_DIM+2, double>;
template class InitialConditionFunction_TurbulentChannelFlow <PHILIP_DIM, PHILIP_DIM+2, double>;
template class InitialConditionFunction_TurbulentChannelFlow_Turbulent <PHILIP_DIM, PHILIP_DIM+2, double>;
template class InitialConditionFunction_TurbulentChannelFlow_Manufactured <PHILIP_DIM, PHILIP_DIM+2, double>;
#endif

#if PHILIP_DIM>1
template class InitialConditionFunction_IsentropicVortex <PHILIP_DIM, PHILIP_DIM+2, double>;
#endif

#if PHILIP_DIM==2
template class InitialConditionFunction_KHI <PHILIP_DIM, PHILIP_DIM+2, double>;
template class InitialConditionFunction_LowDensity2D <PHILIP_DIM, PHILIP_DIM+2, double>;
template class InitialConditionFunction_DoubleMachReflection <PHILIP_DIM, PHILIP_DIM+2, double>;
template class InitialConditionFunction_Mach3WindTunnel <PHILIP_DIM, PHILIP_DIM+2, double>;
template class InitialConditionFunction_SedovBlastWave <PHILIP_DIM, PHILIP_DIM+2, double>;
template class InitialConditionFunction_ShockDiffraction <PHILIP_DIM, PHILIP_DIM+2, double>;
template class InitialConditionFunction_AstrophysicalJet <PHILIP_DIM, PHILIP_DIM+2, double>;
template class InitialConditionFunction_DaruTenaudShockTube <PHILIP_DIM, PHILIP_DIM+2, double>;
template class InitialConditionFunction_DipoleWallCollision <PHILIP_DIM, PHILIP_DIM+2, double>;
template class InitialConditionFunction_DipoleWallCollision_Normal <PHILIP_DIM, PHILIP_DIM+2, double>;
template class InitialConditionFunction_DipoleWallCollision_Oblique <PHILIP_DIM, PHILIP_DIM+2, double>;
#endif

#if PHILIP_DIM < 3
template class InitialConditionFunction_LeblancShockTube<PHILIP_DIM,PHILIP_DIM+2,double>;
#endif

// functions instantiated for all dim
template class InitialConditionFunction_Zero <PHILIP_DIM,1, double>;
template class InitialConditionFunction_Zero <PHILIP_DIM,2, double>;
template class InitialConditionFunction_Zero <PHILIP_DIM,3, double>;
template class InitialConditionFunction_Zero <PHILIP_DIM,4, double>;
template class InitialConditionFunction_Zero <PHILIP_DIM,5, double>;
template class InitialConditionFunction_Zero <PHILIP_DIM,6, double>;
template class InitialConditionFunction_Advection <PHILIP_DIM, 1, double>;
template class InitialConditionFunction_BurgersInviscid <PHILIP_DIM, PHILIP_DIM, double>;
template class InitialConditionFunction_AdvectionEnergy <PHILIP_DIM, 1, double>;
template class InitialConditionFunction_ConvDiff <PHILIP_DIM, 1, double>;
template class InitialConditionFunction_ConvDiffEnergy <PHILIP_DIM,1,double>;
template class InitialConditionFunction_NavierStokesBase <PHILIP_DIM, PHILIP_DIM+2, double>;
template class InitialConditionFunction_SodShockTube <PHILIP_DIM,PHILIP_DIM+2,double>;

} // PHiLiP namespace
