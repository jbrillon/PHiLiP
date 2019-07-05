#include <iostream>

#include <deal.II/grid/grid_out.h>

#include "tests.h"
#include "grid_study.h"
#include "euler_gaussian_bump.h"

namespace PHiLiP {
namespace Tests {

using AllParam = Parameters::AllParameters;

TestsBase::TestsBase(Parameters::AllParameters const *const parameters_input)
    :
    all_parameters(parameters_input)
{}

template<int dim, int nstate>
std::unique_ptr< TestsBase > TestsFactory<dim,nstate>
::select_test(const AllParam *const parameters_input) {
    using Test_enum = AllParam::TestType;
    Test_enum test_type = parameters_input->test_type;

    if(test_type == Test_enum::run_control) {
        return std::make_unique<GridStudy<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::euler_gaussian_bump) {
        if constexpr (dim==2 && nstate==dim+2) return std::make_unique<EulerGaussianBump<dim,nstate>>(parameters_input);
    }

    return nullptr;
}

template<int dim, int nstate>
std::unique_ptr< TestsBase > TestsFactory<dim,nstate>
::create_test(AllParam const *const parameters_input)
{
    // Recursive templating required because template parameters must be compile time constants
    // As a results, this recursive template initializes all possible dimensions with all possible nstate
    // without having 15 different if-else statements
    if(dim == parameters_input->dimension)
    {
        // This template parameters dim and nstate match the runtime parameters
        // then create the selected test with template parameters dim and nstate
        // Otherwise, keep decreasing nstate and dim until it matches
        if(nstate == parameters_input->nstate) 
            return TestsFactory<dim,nstate>::select_test(parameters_input);
        else if constexpr (nstate > 1)
            return TestsFactory<dim,nstate-1>::create_test(parameters_input);
        else
            return nullptr;
    }
    else if constexpr (dim > 1)
    {
        //return TestsFactory<dim-1,nstate>::create_test(parameters_input);
        return nullptr;
    }
    else
    {
        return nullptr;
    }
}

// Will recursively create all the possible test sizes
//template class TestsFactory <PHILIP_DIM,1>;
//template class TestsFactory <PHILIP_DIM,2>;
//template class TestsFactory <PHILIP_DIM,3>;
//template class TestsFactory <PHILIP_DIM,4>;
template class TestsFactory <PHILIP_DIM,5>;

} // Tests namespace
} // PHiLiP namespace
