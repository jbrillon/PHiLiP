set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 1D Viscous Shock Tube - NS
# =======================================

configure_file(1D_viscous_shock_tube.prm 1D_viscous_shock_tube.prm COPYONLY)
add_test(
  NAME 1D_VST_NS_OOA_NSFR_TEST
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_viscous_shock_tube.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)