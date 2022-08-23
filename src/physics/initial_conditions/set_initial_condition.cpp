#include "set_initial_condition.h"
#include <deal.II/numerics/vector_tools.h>
// #include <deal.II/lac/affine_constraints.h>

#include <string>
#include <fstream>


namespace PHiLiP{

template<int dim, int nstate, typename real>
void SetInitialCondition<dim,nstate,real>::set_initial_condition(
        std::shared_ptr< InitialConditionFunction<dim,nstate,double> > initial_condition_function_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real> > dg_input,
        const Parameters::AllParameters *const parameters_input)
{
    // Apply initial condition depending on the application type
    const bool interpolate_initial_condition = parameters_input->flow_solver_param.interpolate_initial_condition;
    if(interpolate_initial_condition == true) {
        // for non-curvilinear
        SetInitialCondition<dim,nstate,real>::interpolate_initial_condition(initial_condition_function_input, dg_input);
    } else {
        // for curvilinear
        SetInitialCondition<dim,nstate,real>::project_initial_condition(initial_condition_function_input, dg_input);
    }
}

template<int dim, int nstate, typename real>
void SetInitialCondition<dim,nstate,real>::interpolate_initial_condition(
        std::shared_ptr< InitialConditionFunction<dim,nstate,double> > &initial_condition_function,
        std::shared_ptr < PHiLiP::DGBase<dim,real> > &dg) 
{
    dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
    solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);
    dealii::VectorTools::interpolate(dg->dof_handler,*initial_condition_function,solution_no_ghost);
    dg->solution = solution_no_ghost;
}

template<int dim, int nstate, typename real>
void SetInitialCondition<dim,nstate,real>::project_initial_condition(
        std::shared_ptr< InitialConditionFunction<dim,nstate,double> > &initial_condition_function,
        std::shared_ptr < PHiLiP::DGBase<dim,real> > &dg) 
{
    // Commented since this has not yet been tested
    // dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
    // solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);
    // dealii::AffineConstraints affine_constraints(dof_handler.locally_owned_dofs());
    // dealii::VectorTools::project(*(dg->high_order_grid->mapping_fe_field),dg->dof_handler,affine_constraints,dg->volume_quadrature_collection,*initial_condition_function,solution_no_ghost);
    // dg->solution = solution_no_ghost;

    //Note that for curvilinear, can't use dealii interpolate since it doesn't project at the correct order.
    //Thus we interpolate it directly.
    const auto mapping = (*(dg->high_order_grid->mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);
    dealii::hp::FEValues<dim,dim> fe_values_collection(mapping_collection, dg->fe_collection, dg->volume_quadrature_collection, 
                                dealii::update_quadrature_points);
    const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    OPERATOR::vol_projection_operator<dim,2*dim> vol_projection(dg->nstate, dg->max_degree, dg->max_grid_degree);
    vol_projection.build_1D_volume_operator(dg->oneD_fe_collection[dg->max_degree], dg->oneD_quadrature_collection[dg->max_degree]);
    for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell) {
        if (!current_cell->is_locally_owned()) continue;
    
        const int i_fele = current_cell->active_fe_index();
        const int i_quad = i_fele;
        const int i_mapp = 0;
        fe_values_collection.reinit (current_cell, i_quad, i_mapp, i_fele);
        const dealii::FEValues<dim,dim> &fe_values = fe_values_collection.get_present_fe_values();
        const unsigned int poly_degree = i_fele;
        const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
        const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
        const unsigned int n_shape_fns = n_dofs_cell/nstate;
        current_dofs_indices.resize(n_dofs_cell);
        current_cell->get_dof_indices (current_dofs_indices);
        for(int istate=0; istate<nstate; istate++){
            std::vector<double> exact_value(n_quad_pts);
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                const dealii::Point<dim> qpoint = (fe_values.quadrature_point(iquad));
                exact_value[iquad] = initial_condition_function->value(qpoint, istate);
            }   
            std::vector<double> sol(n_shape_fns);
            vol_projection.matrix_vector_mult_1D(exact_value, sol, vol_projection.oneD_vol_operator);
            for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                dg->solution[current_dofs_indices[ishape+istate*n_shape_fns]] = sol[ishape];
            }
        }
    }
}

void add_value_to_data_table(
    const double value,
    const std::string value_string,
    std::shared_ptr <dealii::TableHandler> data_table)
{
    data_table->add_value(value_string, value);
    data_table->set_precision(value_string, 16);
    data_table->set_scientific(value_string, true);
}

template<int dim, int nstate, typename real>
void SetInitialCondition<dim,nstate,real>::project_initial_condition_from_file(
        std::shared_ptr < PHiLiP::DGBase<dim,real> > &dg,
        std::shared_ptr <dealii::TableHandler> unsteady_data_table) 
{
    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    // //Note that for curvilinear, can't use dealii interpolate since it doesn't project at the correct order.
    // //Thus we interpolate it directly.
    // const auto mapping = (*(dg->high_order_grid->mapping_fe_field));
    // dealii::hp::MappingCollection<dim> mapping_collection(mapping);
    // dealii::hp::FEValues<dim,dim> fe_values_collection(mapping_collection, dg->fe_collection, dg->volume_quadrature_collection, 
    //                             dealii::update_quadrature_points);
    // const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    // std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    // OPERATOR::vol_projection_operator<dim,2*dim> vol_projection(dg->nstate, dg->max_degree, dg->max_grid_degree);
    // vol_projection.build_1D_volume_operator(dg->oneD_fe_collection[dg->max_degree], dg->oneD_quadrature_collection[dg->max_degree]);
    
    // for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell) {
    //     if (!current_cell->is_locally_owned()) continue;
    
    //     const int i_fele = current_cell->active_fe_index();
    //     const int i_quad = i_fele;
    //     const int i_mapp = 0;
    //     fe_values_collection.reinit (current_cell, i_quad, i_mapp, i_fele);
    //     const dealii::FEValues<dim,dim> &fe_values = fe_values_collection.get_present_fe_values();
    //     const unsigned int poly_degree = i_fele;
    //     const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
    //     const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    //     const unsigned int n_shape_fns = n_dofs_cell/nstate;
    //     current_dofs_indices.resize(n_dofs_cell);
    //     current_cell->get_dof_indices (current_dofs_indices);
    //     if(mpi_rank==0) std::cout << "n_quad_pts = " << n_quad_pts << std::endl;
    //     for(int istate=0; istate<1; istate++){
    //         std::vector<double> exact_value(n_quad_pts);
    //         // exact_value.fill(0.0);
    //         for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
    //             const dealii::Point<dim> qpoint = (fe_values.quadrature_point(iquad));
    //             // if(mpi_rank==0) {
    //             //     add_value_to_data_table(qpoint[0],"x",unsteady_data_table);
    //             //     if constexpr (dim>1) add_value_to_data_table(qpoint[1],"y",unsteady_data_table);
    //             //     if constexpr (dim>2) add_value_to_data_table(qpoint[2],"z",unsteady_data_table);
    //             // }
    //             // exact_value[iquad] = initial_condition_function->value(qpoint, istate);
    //             exact_value[iquad] = qpoint[0];
    //         }   
    //         std::vector<double> sol(n_shape_fns);
    //         vol_projection.matrix_vector_mult_1D(exact_value, sol, vol_projection.oneD_vol_operator);
    //         for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
    //             dg->solution[current_dofs_indices[ishape+istate*n_shape_fns]] = sol[ishape];
    //         }
    //     }
    // }

    // TAKEN FROM PERIODIC TURBULENCE BRANCH
    int overintegrate = 0;
    dealii::QGaussLobatto<dim> quad_extra(dg->max_degree+1+overintegrate);
    dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[dg->max_degree], quad_extra,
                                              dealii::update_values | dealii::update_gradients | dealii::update_JxW_values | dealii::update_quadrature_points);

    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    // std::array<double,nstate> soln_at_q;
    // std::array<dealii::Tensor<1,dim,double>,nstate> soln_grad_at_q;

    std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
    // for (auto cell : dg.dof_handler.active_cell_iterators()) {
    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;
        fe_values_extra.reinit (cell);
        cell->get_dof_indices (dofs_indices);
        for(int istate=0; istate<1; istate++){
            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));
                // if(mpi_rank==0) {
                    add_value_to_data_table(qpoint[0],"x",unsteady_data_table);
                    if constexpr (dim>1) add_value_to_data_table(qpoint[1],"y",unsteady_data_table);
                    if constexpr (dim>2) add_value_to_data_table(qpoint[2],"z",unsteady_data_table);
                    // add_value_to_data_table(istate,"state",unsteady_data_table);
                // }
            }
        }
    }

    // if(mpi_rank==0) {
        std::string restart_unsteady_data_table_filename = std::string("coord_check-proc_")+std::to_string(mpi_rank)+std::string(".txt");
        std::ofstream unsteady_data_table_file(restart_unsteady_data_table_filename);
        unsteady_data_table->write_text(unsteady_data_table_file);
    // }
}

template class SetInitialCondition<PHILIP_DIM, 1, double>;
template class SetInitialCondition<PHILIP_DIM, 2, double>;
template class SetInitialCondition<PHILIP_DIM, 3, double>;
template class SetInitialCondition<PHILIP_DIM, 4, double>;
template class SetInitialCondition<PHILIP_DIM, 5, double>;

// template class SetInitialCondition<PHILIP_DIM, 1, double>::project_initial_condition_from_file(std::shared_ptr < PHiLiP::DGBase<PHILIP_DIM,double> > dg, std::shared_ptr <dealii::TableHandler> /*unsteady_data_table*/);
// template class SetInitialCondition<PHILIP_DIM, 2, double>::project_initial_condition_from_file(std::shared_ptr < PHiLiP::DGBase<PHILIP_DIM,double> > dg, std::shared_ptr <dealii::TableHandler> /*unsteady_data_table*/);
// template class SetInitialCondition<PHILIP_DIM, 3, double>::project_initial_condition_from_file(std::shared_ptr < PHiLiP::DGBase<PHILIP_DIM,double> > dg, std::shared_ptr <dealii::TableHandler> /*unsteady_data_table*/);
// template class SetInitialCondition<PHILIP_DIM, 4, double>::project_initial_condition_from_file(std::shared_ptr < PHiLiP::DGBase<PHILIP_DIM,double> > dg, std::shared_ptr <dealii::TableHandler> /*unsteady_data_table*/);
// template class SetInitialCondition<PHILIP_DIM, 5, double>::project_initial_condition_from_file(std::shared_ptr < PHiLiP::DGBase<PHILIP_DIM,double> > dg, std::shared_ptr <dealii::TableHandler> /*unsteady_data_table*/);

}//end of namespace PHILIP
