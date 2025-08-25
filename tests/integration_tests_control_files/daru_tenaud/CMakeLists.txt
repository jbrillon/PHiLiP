set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 2D Daru Tenaud
# =======================================
configure_file(2D_daru_tenaud.prm 2D_daru_tenaud.prm COPYONLY)
add_test(
  NAME 2D_DARU_TENAUD
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_daru_tenaud.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

