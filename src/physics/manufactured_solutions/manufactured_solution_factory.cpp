#include "ADTypes.hpp"

#include "manufactured_solution_factory.h"

namespace PHiLiP {

template <int dim, int nstate, typename real>
std::shared_ptr < ManufacturedSolutionFunction<dim,nstate,real> >
ManufacturedSolutionFactory<dim,nstate,real>
::create_ManufacturedSolution(const Parameters::ManufacturedConvergenceStudyParam::ManufacturedSolutionType manu_sol_type_input)
{
	// MST abbreviated ManufacturedSolutionType
	using MST_enum = Parameters::ManufacturedConvergenceStudyParam::ManufacturedSolutionType;

	MST_enum manufactured_solution_type = manu_sol_type_input;

	if (manufactured_solution_type == MST_enum::sine){
		return std::make_shared < Sine<dim,nstate,real> >();
	} else if (manufactured_solution_type == MST_enum::additive){
		return std::make_shared < Additive<dim,nstate,real> >();
	} else if (manufactured_solution_type == MST_enum::cosine){
		return std::make_shared < Cosine<dim,nstate,real> >();
	} else if (manufactured_solution_type == MST_enum::arctangent){
		return std::make_shared < Arctangent<dim,nstate,real> >();
	} else if (manufactured_solution_type == MST_enum::exponential){
		return std::make_shared < Exponential<dim,nstate,real> >();
	} else if (manufactured_solution_type == MST_enum::even_polynomial){
		return std::make_shared < Even_Polynomial<dim,nstate,real> >();
	} else if (manufactured_solution_type == MST_enum::polynomial){
		return std::make_shared < Polynomial<dim,nstate,real> >();
	}
	std::cout << "Can't create ManufacturedSolutionFunction, invalid manufactured solution type: " << manufactured_solution_type << std::endl;
    assert(0==1 && "Can't create ManufacturedSolutionFunction, invalid manufactured solution type");
    return nullptr;
}

template class ManufacturedSolutionFactory<PHILIP_DIM, 1, double>;
template class ManufacturedSolutionFactory<PHILIP_DIM, 2, double>;
template class ManufacturedSolutionFactory<PHILIP_DIM, 3, double>;
template class ManufacturedSolutionFactory<PHILIP_DIM, 4, double>;
template class ManufacturedSolutionFactory<PHILIP_DIM, 5, double>;
template class ManufacturedSolutionFactory<PHILIP_DIM, 8, double>;

template class ManufacturedSolutionFactory<PHILIP_DIM, 1, FadType>;
template class ManufacturedSolutionFactory<PHILIP_DIM, 2, FadType>;
template class ManufacturedSolutionFactory<PHILIP_DIM, 3, FadType>;
template class ManufacturedSolutionFactory<PHILIP_DIM, 4, FadType>;
template class ManufacturedSolutionFactory<PHILIP_DIM, 5, FadType>;
template class ManufacturedSolutionFactory<PHILIP_DIM, 8, FadType>;

template class ManufacturedSolutionFactory<PHILIP_DIM, 1, RadType>;
template class ManufacturedSolutionFactory<PHILIP_DIM, 2, RadType>;
template class ManufacturedSolutionFactory<PHILIP_DIM, 3, RadType>;
template class ManufacturedSolutionFactory<PHILIP_DIM, 4, RadType>;
template class ManufacturedSolutionFactory<PHILIP_DIM, 5, RadType>;
template class ManufacturedSolutionFactory<PHILIP_DIM, 8, RadType>;

template class ManufacturedSolutionFactory<PHILIP_DIM, 1, FadFadType>;
template class ManufacturedSolutionFactory<PHILIP_DIM, 2, FadFadType>;
template class ManufacturedSolutionFactory<PHILIP_DIM, 3, FadFadType>;
template class ManufacturedSolutionFactory<PHILIP_DIM, 4, FadFadType>;
template class ManufacturedSolutionFactory<PHILIP_DIM, 5, FadFadType>;
template class ManufacturedSolutionFactory<PHILIP_DIM, 8, FadFadType>;

template class ManufacturedSolutionFactory<PHILIP_DIM, 1, RadFadType>;
template class ManufacturedSolutionFactory<PHILIP_DIM, 2, RadFadType>;
template class ManufacturedSolutionFactory<PHILIP_DIM, 3, RadFadType>;
template class ManufacturedSolutionFactory<PHILIP_DIM, 4, RadFadType>;
template class ManufacturedSolutionFactory<PHILIP_DIM, 5, RadFadType>;
template class ManufacturedSolutionFactory<PHILIP_DIM, 8, RadFadType>;


} // PHiLiP namespace