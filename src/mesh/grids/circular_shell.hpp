#ifndef __CIRCULAR_SHELL_H__
#define __CIRCULAR_SHELL_H__

#include <deal.II/distributed/tria.h>

namespace PHiLiP {
namespace Grids {

/** Create a circular shell with wall and Dirichlet boundary conditions
 *  on outer and inner boundaries, respectively. 
 * */
template<int dim, typename TriangulationType>
void circular_shell(std::shared_ptr<TriangulationType> &grid,
                    const double domain_inner_radius,
                    const double domain_outer_radius,
                    const int number_of_cells_in_radial_direction);

} // namespace Grids
} // namespace PHiLiP
#endif

