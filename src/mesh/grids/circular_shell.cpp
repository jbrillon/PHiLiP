#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/grid/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <stdlib.h>
#include <iostream>

#include "circular_shell.hpp"

namespace PHiLiP {
namespace Grids {

template<int dim, typename TriangulationType>
void circular_shell(std::shared_ptr<TriangulationType> &grid,
                    const double domain_inner_radius,
                    const double domain_outer_radius,
                    const int number_of_cells_in_radial_direction)
{
    const bool colorize = true;
    dealii::Point<dim> origin;
    for(int d=0; d<dim; ++d) {
        origin[d] = 0.0;
    }
    if constexpr(dim==2) {
        dealii::GridGenerator::hyper_shell(*grid, origin, domain_inner_radius, domain_outer_radius, number_of_cells_in_radial_direction, colorize);
        for (auto cell = grid->begin_active(); cell != grid->end(); ++cell) {
            for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
                auto current_face = cell->face(face);
                if (current_face->at_boundary()){
                    if(current_face->boundary_id() == 0) {
                        // inner boundary
                        current_face->set_boundary_id (1007); // Circular-Couette flow Dirichlet boundary condition
                    } else if (current_face->boundary_id() == 1) {
                        // outer boundary
                        current_face->set_boundary_id (1001); // wall boundary condition
                    } // else: do nothing
                }
            }
        }
    } else {
        std::cout << "ERROR: circular_shell only implemented for 2D. Aborting..." << std::endl;
        std::abort();
    }
}

#if PHILIP_DIM==2
    template void circular_shell<PHILIP_DIM, dealii::parallel::distributed::Triangulation<PHILIP_DIM>> (std::shared_ptr<dealii::parallel::distributed::Triangulation<PHILIP_DIM>> &grid, const double domain_left, const double domain_right, const int number_of_cells_per_direction);
#endif

} // namespace Grids
} // namespace PHiLiP
