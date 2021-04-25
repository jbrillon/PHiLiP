#ifndef __SIZE_FIELD_H__
#define __SIZE_FIELD_H__

#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>

#include "grid_refinement/field.h"

namespace PHiLiP {

namespace GridRefinement {

// wrapper for the set of functions to determine mesh size density functions
// based on input of spatial error data (and possibly anisotropy)
template <int dim, typename real>
class SizeField
{
public:
    // computes the isotropic size field (h has only 1 component) for a uniform (p-dist constant) input
    static void isotropic_uniform(
        const real &                               complexity,   // (input) complexity target
        const dealii::Vector<real> &               B,            // only one since p is constant
        const dealii::hp::DoFHandler<dim> &        dof_handler,  // dof_handler
        std::unique_ptr<Field<dim,real>> &         h_field,      // (output) size field
        const real &                               poly_degree); // (input)  polynomial degree

    // computes isotropic size field (h has only 1 component) for a non-uniform p_field
    static void isotropic_h(
        const real                                 complexity,            // (input) complexity target
        const dealii::Vector<real> &               B,                     // only one since p is constant
        const dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
        const dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
        const dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
        std::unique_ptr<Field<dim,real>> &         h_field,               // (output) size field
        const dealii::Vector<real> &               p_field);              // (input)  poly field

    // computes updated p-field with a constant h-field
    static void isotropic_p(
        const dealii::Vector<real> &               Bm,                    // constant for p-1
        const dealii::Vector<real> &               B,                     // constant for p
        const dealii::Vector<real> &               Bp,                    // constant for p+1
        const dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
        const dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
        const dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
        const std::unique_ptr<Field<dim,real>> &   h_field,               // (input) size field
        dealii::Vector<real> &                     p_field);              // (output) poly field

    // updates both the h-field and p-field
    static void isotropic_hp(
        const real                                 complexity,            // target complexity
        const dealii::Vector<real> &               Bm,                    // constant for p-1
        const dealii::Vector<real> &               B,                     // constant for p
        const dealii::Vector<real> &               Bp,                    // constant for p+1
        const dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
        const dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
        const dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
        std::unique_ptr<Field<dim,real>> &         h_field,               // (output) size field
        dealii::Vector<real> &                     p_field);              // (output) poly field

    // performs adjoint based size-field adaptation with uniform p-field. Based on:
    // Balan, A., Woopen, M., & May, G. (2016). 
    // Adjoint-based hp-adaptivity on anisotropic meshes for high-order compressible flow simulations. 
    // Computers and Fluids, 139. https://doi.org/10.1016/j.compfluid.2016.03.029
    static void adjoint_uniform_balan(
        const real                                 complexity,            // target complexity
        const real                                 r_max,                 // maximum refinement factor
        const real                                 c_max,                 // maximum coarsening factor
        const dealii::Vector<real> &               eta,                   // error indicator (DWR)
        const dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
        const dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
        const dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
        std::unique_ptr<Field<dim,real>>&          h_field,               // (output) target size_field
        const real &                               poly_degree);          // uniform polynomial degree

    // performs adjoint based size-field adaptation with general p-field. Based on:
    // Balan, A., Woopen, M., & May, G. (2016). 
    // Adjoint-based hp-adaptivity on anisotropic meshes for high-order compressible flow simulations. 
    // Computers and Fluids, 139. https://doi.org/10.1016/j.compfluid.2016.03.029
    static void adjoint_h_balan(
        const real                                 complexity,            // target complexity
        const real                                 r_max,                 // maximum refinement factor
        const real                                 c_max,                 // maximum coarsening factor
        const dealii::Vector<real> &               eta,                   // error indicator (DWR)
        const dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
        const dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
        const dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
        std::unique_ptr<Field<dim,real>>&          h_field,               // (output) target size_field
        const dealii::Vector<real> &               p_field);              // polynomial degree vector

    // performs adjoint based size field adaptatation with uniform p-field
    // peforms equidistribution of DWR to sizes based on 2p+1 power of convergence
    static void adjoint_h_equal(
        const real                                 complexity,            // target complexity
        const dealii::Vector<real> &               eta,                   // error indicator (DWR)
        const dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
        const dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
        const dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
        std::unique_ptr<Field<dim,real>>&          h_field,               // (output) target size_field
        const real &                               poly_degree);          // uniform polynomial degree

protected:
    // sets the h_field sizes based on a reference value and DWR distribution
    static void update_h_dwr(
        const real                          tau,          // reference value for settings sizes
        const dealii::Vector<real> &        eta,          // error indicator (DWR)
        const dealii::hp::DoFHandler<dim> & dof_handler,  // dof_handler
        std::unique_ptr<Field<dim,real>>&   h_field,      // (output) target size_field
        const real &                        poly_degree); // uniform polynomial degree

    // given a p-field, redistribute h_field according to B
    static void update_h_optimal(
        const real                          lambda,      // (input) bisection parameter
        const dealii::Vector<real> &        B,           // constant for current p
        const dealii::hp::DoFHandler<dim> & dof_handler, // dof_handler
        std::unique_ptr<Field<dim,real>> &  h_field,     // (output) size field
        const dealii::Vector<real> &        p_field);    // (input)  poly field

    static real evaluate_complexity(
        const dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
        const dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
        const dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
        const std::unique_ptr<Field<dim,real>> &   h_field,               // (input) size field
        const dealii::Vector<real> &               p_field);              // (input) poly field  

    static void update_alpha_vector_balan(
        const dealii::Vector<real>&        eta,         // vector of DWR indicators
        const real                         r_max,       // max refinement factor
        const real                         c_max,       // max coarsening factor
        const real                         eta_min,     // minimum DWR
        const real                         eta_max,     // maximum DWR
        const real                         eta_ref,     // reference parameter for bisection
        const dealii::hp::DoFHandler<dim>& dof_handler, // dof_handler
        const dealii::Vector<real>&        I_c,         // cell area measure
        std::unique_ptr<Field<dim,real>>&  h_field);    // (output) size-field

    static real update_alpha_k_balan(
        const real eta_k,   // local DWR factor
        const real r_max,   // maximum refinement factor
        const real c_max,   // maximum coarsening factor
        const real eta_min, // minimum DWR indicator
        const real eta_max, // maximum DWR indicator
        const real eta_ref);// referebce DWR for determining coarsening/refinement

    static real bisection(
        const std::function<real(real)> func,         // lambda function that takes real -> real 
        real                            lower_bound,  // lower bound of the search
        real                            upper_bound,  // upper bound of the search
        real                            rel_tolerance = 1e-6,
        real                            abs_tolerance = 1.0);
};

} // namespace GridRefinement

} //namespace PHiLiP

#endif // __SIZE_FIELD_H__
