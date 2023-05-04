#ifndef __FLOW_SOLVER_METRIC_FIELD_EVALUATION__
#define __FLOW_SOLVER_METRIC_FIELD_EVALUATION__

#include "tests.h"

namespace PHiLiP {
namespace Tests {

/// Flow Solver Metric Field Evaluation
template <int dim, int nstate>
class FlowSolverMetricFieldEvaluation: public TestsBase
{
public:
    /// Constructor
    FlowSolverMetricFieldEvaluation(
        const Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input);

    /// Destructor
    ~FlowSolverMetricFieldEvaluation() {};

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;

    /// Run test
    int run_test () const override;
};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
