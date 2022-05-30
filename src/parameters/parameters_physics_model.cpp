#include "parameters/parameters_physics_model.h"

namespace PHiLiP {
namespace Parameters {
    
// Models inputs
PhysicsModelParam::PhysicsModelParam () {}

void PhysicsModelParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("physics_model");
    {
        prm.enter_subsection("large_eddy_simulation");
        {
            prm.declare_entry("euler_turbulence","false",
                              dealii::Patterns::Bool(),
                              "Set as false by default. If true, sets the baseline physics for LES to the Euler equations.");

            prm.declare_entry("SGS_model_type", "smagorinsky",
                              dealii::Patterns::Selection(
                              " smagorinsky | "
                              " wall_adaptive_local_eddy_viscosity | "
                              " vreman"),
                              "Enum of sub-grid scale models."
                              "Choices are "
                              " <smagorinsky | "
                              "  wall_adaptive_local_eddy_viscosity | "
                              "  vreman>.");

            prm.declare_entry("turbulent_prandtl_number", "0.6",
                              dealii::Patterns::Double(1e-15, 10000000),
                              "Turbulent Prandlt number (default is 0.6)");

            prm.declare_entry("smagorinsky_model_constant", "0.1",
                              dealii::Patterns::Double(1e-15, 10000000),
                              "Smagorinsky model constant (default is 0.1)");

            prm.declare_entry("WALE_model_constant", "0.6",
                              dealii::Patterns::Double(1e-15, 0.6),
                              "WALE (Wall-Adapting Local Eddy-viscosity) eddy viscosity model constant (default is 0.6)");

            prm.declare_entry("vreman_model_constant", "0.025",
                              dealii::Patterns::Double(1e-15, 0.6),
                              "Vreman eddy viscosity model constant (default is 0.025)");
            
            prm.declare_entry("ratio_of_filter_width_to_cell_size", "1.0",
                              dealii::Patterns::Double(1e-15, 10000000),
                              "Ratio of the large eddy simulation filter width to the cell size (default is 1)");

        }
        prm.leave_subsection();
    }
    prm.leave_subsection();
}

void PhysicsModelParam::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("physics_model");
    {
        prm.enter_subsection("large_eddy_simulation");
        {
            euler_turbulence = prm.get_bool("euler_turbulence");

            const std::string SGS_model_type_string = prm.get("SGS_model_type");
            if(SGS_model_type_string == "smagorinsky")                        SGS_model_type = smagorinsky;
            if(SGS_model_type_string == "wall_adaptive_local_eddy_viscosity") SGS_model_type = wall_adaptive_local_eddy_viscosity;
            if(SGS_model_type_string == "vreman")                             SGS_model_type = vreman;

            turbulent_prandtl_number           = prm.get_double("turbulent_prandtl_number");
            smagorinsky_model_constant         = prm.get_double("smagorinsky_model_constant");
            WALE_model_constant                = prm.get_double("WALE_model_constant");
            vreman_model_constant              = prm.get_double("vreman_model_constant");
            ratio_of_filter_width_to_cell_size = prm.get_double("ratio_of_filter_width_to_cell_size");
        }
        prm.leave_subsection();
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace
