#ifndef __GRID_REFINEMENT_CONTINUOUS_H__
#define __GRID_REFINEMENT_CONTINUOUS_H__

#include "grid_refinement/grid_refinement.h"

namespace PHiLiP {

namespace GridRefinement {

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class GridRefinement_Continuous : public GridRefinementBase<dim,nstate,real,MeshType>
{
public:
    
    // deleting the default constructor
    GridRefinement_Continuous() = delete;

    // overriding the other constructors to call this delegated constructors constructors
    GridRefinement_Continuous(
        PHiLiP::Parameters::GridRefinementParam                          gr_param_input,
        std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real, MeshType> >  adj_input,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input);

    GridRefinement_Continuous(
        PHiLiP::Parameters::GridRefinementParam                            gr_param_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> >             dg_input,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> >   physics_input,
        std::shared_ptr< PHiLiP::Functional<dim, nstate, real, MeshType> > functional_input);

    GridRefinement_Continuous(
        PHiLiP::Parameters::GridRefinementParam                          gr_param_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> >           dg_input,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input);

    GridRefinement_Continuous(
        PHiLiP::Parameters::GridRefinementParam                gr_param_input,
        // PHiLiP::Parameters::AllParameters const *const param_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> > dg_input);

protected:
    // delegated constructor
    GridRefinement_Continuous(
        PHiLiP::Parameters::GridRefinementParam                            gr_param_input,
        std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real, MeshType> >    adj_input,
        std::shared_ptr< PHiLiP::Functional<dim, nstate, real, MeshType> > functional_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> >             dg_input,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> >   physics_input);

    using GridRefinementBase<dim,nstate,real,MeshType>::GridRefinementBase;

    // virtual refinement method from base class
    void refine_grid() override;

protected:

    // specified refinement functions for different cases
    void refine_grid_h();
    void refine_grid_p();
    void refine_grid_hp();    

    // vtk output function
    std::vector< std::pair<dealii::Vector<real>, std::string> > output_results_vtk_method() override;

    // getting the size or tensor fields based on indicator
    void field();
    virtual void field_h() = 0;
    virtual void field_p() = 0;
    virtual void field_hp() = 0;
    
    // performing output to appropriate mesh generator
    void refine_grid_gmsh();
    void refine_grid_msh();

    // scheduling of complexity growth
    real current_complexity();
    void target_complexity();

    real              complexity_initial;
    real              complexity_target;
    std::vector<real> complexity_vector;

    void get_current_field_h();
    void get_current_field_p();

    std::unique_ptr<Field<dim,real>> h_field;
    dealii::Vector<real> p_field;
};

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class GridRefinement_Continuous_Error : public GridRefinement_Continuous<dim,nstate,real,MeshType>
{
public:
    using GridRefinement_Continuous<dim,nstate,real,MeshType>::GridRefinement_Continuous;
    void field_h()  override;
    void field_p()  override;
    void field_hp() override;
};

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class GridRefinement_Continuous_Hessian : public GridRefinement_Continuous<dim,nstate,real,MeshType>
{
public:
    using GridRefinement_Continuous<dim,nstate,real,MeshType>::GridRefinement_Continuous;
    void field_h()  override;
    void field_p()  override;
    void field_hp() override;
};

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class GridRefinement_Continuous_Residual : public GridRefinement_Continuous<dim,nstate,real,MeshType>
{
public:
    using GridRefinement_Continuous<dim,nstate,real,MeshType>::GridRefinement_Continuous;
    void field_h()  override;
    void field_p()  override;
    void field_hp() override;
};

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class GridRefinement_Continuous_Adjoint : public GridRefinement_Continuous<dim,nstate,real,MeshType>
{
public:
    using GridRefinement_Continuous<dim,nstate,real,MeshType>::GridRefinement_Continuous;
    void field_h()  override;
    void field_p()  override;
    void field_hp() override;
};

} // namespace GridRefinement

} // namespace PHiLiP

#endif // __GRID_REFINEMENT_CONTINUOUS_H__