#include "flow_solver_metric_field_evaluation.h"
#include "flow_solver/flow_solver_factory.h"

#include <deal.II/grid/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>
#include <Sacado.hpp>
#include "tests.h"
#include "grid_refinement_study.h"
#include "functional/functional.h"
#include "functional/adjoint.h"
#include "grid_refinement/grid_refinement.h"
#include "physics/physics_factory.h"
#include "physics/model_factory.h"

namespace PHiLiP {
namespace Tests {

#if PHILIP_DIM==1
    using MeshType = dealii::Triangulation<PHILIP_DIM>;
#else
    using MeshType = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

template <int dim, int nstate>
FlowSolverMetricFieldEvaluation<dim, nstate>::FlowSolverMetricFieldEvaluation(
    const PHiLiP::Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
int FlowSolverMetricFieldEvaluation<dim, nstate>::run_test() const
{
    // Run flow solver to get solution
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(this->all_parameters, parameter_handler);
    static_cast<void>(flow_solver->run());

    // get grs_param and gr_param
    const Parameters::AllParameters param                = *(TestsBase::all_parameters);
    const Parameters::GridRefinementStudyParam grs_param = param.grid_refinement_study_param;
    const Parameters::GridRefinementParam gr_param = grs_param.grid_refinement_param_vector[0]; // Note using zero here
    // const unsigned int refinement_steps = gr_param.refinement_steps; // set to 1 in prm file

    // generate Functional
    std::shared_ptr< Functional<dim,nstate,double,MeshType> > functional 
        = FunctionalFactory<dim,nstate,double,MeshType>::create_Functional(grs_param.functional_param, flow_solver->dg);

    using ADtype = Sacado::Fad::DFad<double>;

    // const unsigned int poly_degree      = grs_param.poly_degree;
    // const unsigned int poly_degree_max  = grs_param.poly_degree_max;
    // const unsigned int poly_degree_grid = grs_param.poly_degree_grid;

    // const unsigned int num_refinements = grs_param.num_refinements;

    // generate Adjoint:
    // - creating the model object for physics
    std::shared_ptr< Physics::ModelBase<dim,nstate,double> > model_double
        = Physics::ModelFactory<dim,nstate,double>::create_Model(&param);
    std::shared_ptr< Physics::ModelBase<dim,nstate,ADtype> > model_adtype
        = Physics::ModelFactory<dim,nstate,ADtype>::create_Model(&param);
    // - creating the physics object
    std::shared_ptr< Physics::PhysicsBase<dim,nstate,double> > physics_double
        = Physics::PhysicsFactory<dim,nstate,double>::create_Physics(&param,model_double);
    std::shared_ptr< Physics::PhysicsBase<dim,nstate,ADtype> > physics_adtype
        = Physics::PhysicsFactory<dim,nstate,ADtype>::create_Physics(&param,model_adtype);    
    // --> generate Adjoint
    std::shared_ptr< Adjoint<dim,nstate,double,MeshType> > adjoint 
        = std::make_shared< Adjoint<dim,nstate,double,MeshType> >(flow_solver->dg, functional, physics_adtype);

    // generate the GridRefinement
    std::shared_ptr< GridRefinement::GridRefinementBase<dim,nstate,double,MeshType> > grid_refinement 
        = GridRefinement::GridRefinementFactory<dim,nstate,double,MeshType>::create_GridRefinement(gr_param,adjoint,physics_double);

    // takes care of everything
    grid_refinement->refine_grid();

    return 0;
}

#if PHILIP_DIM==2
    template class FlowSolverMetricFieldEvaluation<PHILIP_DIM,PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace