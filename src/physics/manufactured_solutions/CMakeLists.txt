SET(SOURCE
    manufactured_solution.cpp
    additive.cpp
    arctangent.cpp
    cosine.cpp
    even_polynomial.cpp
    exponential.cpp
    polynomial.cpp
    sine.cpp
    manufactured_solution_factory.cpp
    )

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT ManufacturedSolutionsLib ManufacturedSolutions_${dim}D)
    add_library(${ManufacturedSolutionsLib} STATIC ${SOURCE})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${ManufacturedSolutionsLib})
    endif()

    target_compile_definitions(${ManufacturedSolutionsLib} PRIVATE PHILIP_DIM=${dim})

    # Setup target with deal.II
    unset(ManufacturedSolutionsLib)

endforeach()
