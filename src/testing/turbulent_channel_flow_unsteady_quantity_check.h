#ifndef __TURBULENT_CHANNEL_FLOW_UNSTEADY_QUANTITY_CHECK__
#define __TURBULENT_CHANNEL_FLOW_UNSTEADY_QUANTITY_CHECK__

#include "tests.h"

namespace PHiLiP {
namespace Tests {

/// Turbulent Channel Flow Unsteady Quantity Check
template <int dim, int nstate>
class TurbulentChannelFlowUnsteadyQuantityCheck: public TestsBase
{
public:
    /// Constructor
    TurbulentChannelFlowUnsteadyQuantityCheck(
        const Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input);

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;
    
    /// Expected kinetic energy at final time
    const double kinetic_energy_expected;

    /// Expected enstrophy at final time
    const double enstrophy_expected;

    /// Expected palinstrophy at final time
    const double palinstrophy_expected;

    /// Run test
    int run_test () const override;
};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
