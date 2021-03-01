#include "ADTypes.hpp"

#include "manufactured_solution_factory.h"

namespace PHiLiP {

template <int dim, typename real>
std::shared_ptr < ManufacturedSolutionFunction<dim,real> >
ManufacturedSolutionFactory<dim,real>
::create_ManufacturedSolution(const Parameters::ManufacturedConvergenceStudyParam *const parameters_input)
{
	// MST abbreviated ManufacturedSolutionType
	using MST_enum = Parameters::ManufacturedConvergenceStudyParam::ManufacturedSolutionType;

	MST_enum manufactured_solution_type = parameters_input->manufactured_solution_type;

	if (manufactured_solution_type == MST_enum::sine){
		return std::make_shared < Sine<dim,real>;
	} else if (manufactured_solution_type == MST_enum::additive){
		return std::make_shared < Additive<dim,real>;
	} else if (manufactured_solution_type == MST_enum::cosine){
		return std::make_shared < Cosine<dim,real>;
	} else if (manufactured_solution_type == MST_enum::arctangent){
		return std::make_shared < Arctangent<dim,real>;
	} else if (manufactured_solution_type == MST_enum::exponential){
		return std::make_shared < Exponential<dim,real>;
	} else if (manufactured_solution_type == MST_enum::even_polynomial){
		return std::make_shared < Even_Polynomial<dim,real>;
	} else if (manufactured_solution_type == MST_enum::polynomial){
		return std::make_shared < Polynomial<dim,real>;
	}
	std::cout << "Can't create ManufacturedSolutionFunction, invalid manufactured solution type: " << manufactured_solution_type << std::endl;
    assert(0==1 && "Can't create ManufacturedSolutionFunction, invalid manufactured solution type");
    return nullptr;
}

template class ManufacturedSolutionFactory<PHILIP_DIM, double>;
template class ManufacturedSolutionFactory<PHILIP_DIM, FadType >;
template class ManufacturedSolutionFactory<PHILIP_DIM, RadType >;
template class ManufacturedSolutionFactory<PHILIP_DIM, FadFadType >;
template class ManufacturedSolutionFactory<PHILIP_DIM, RadFadType >;


} // PHiLiP namespace