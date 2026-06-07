cmake_minimum_required(VERSION 3.16)

###############################################################################
# Application related definitions
###############################################################################
set(TARGET_NAME         tdm_xxs_single_ce_224_best)
set(MAIN                tdm_xxs_single_ce_224_best.c)


set(AT_MODEL_PATH       ${CMAKE_CURRENT_SOURCE_DIR}/at_model/model.c)
set(AT_TENSORS_DIR      ${CMAKE_CURRENT_SOURCE_DIR}/tensors)
set(MODEL_EXPRESSIONS   ${CMAKE_CURRENT_SOURCE_DIR}/at_model/Expression_Kernels.c)


###############################################################################
# CMake pre initialization
###############################################################################
include($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)

###############################################################################
# Autotiler Project Setup
###############################################################################
autotiler_add_model(
  TARGET_NAME tdm_xxs_single_ce_224_best
  MODEL_PATH ${AT_MODEL_PATH}
  MODEL_TENSORS_DIR ${AT_TENSORS_DIR}
  PRI1_USED TRUE
  PRI1_FLASH mram
  PRI1_FILENAME "L3_PrivilegedFlash_Const.dat"
  PRI2_USED TRUE
  PRI2_FLASH flash
  PRI2_FILENAME "L3_Flash_Const.dat"
  L1_MEMORY ${CONFIG_MODEL_L1_MEMORY}
  L2_MEMORY ${CONFIG_MODEL_L2_MEMORY}
  L3_MEMORY ${CONFIG_MODEL_L3_MEMORY}
  L2STATIC_MEMORY ${CONFIG_MODEL_L2STATIC_MEMORY}
  L3STATIC_MEMORY ${CONFIG_MODEL_L3STATIC_MEMORY})

###############################################################################
# Application dependencies
###############################################################################
list(TRANSFORM KERNEL_C PREPEND ${CONFIG_MODEL_BUILDDIR}/)
list(APPEND TARGET_SRCS ${MAIN}
                        ${CNN_LIB}
                        ${KERNEL_C}
                        ${MODEL_EXPRESSIONS}
)

list(APPEND TARGET_INCS ${CMAKE_CURRENT_SOURCE_DIR}
                        ${TILER_INC}
                        ${TILER_EMU_INC}
                        $ENV{GAP_SDK_HOME}/libs/gap_lib/include
                        $ENV{GAP_SDK_HOME}/utils/power_meas_utils
                        ${CNN_LIB_INCLUDE_NOPREPEND}
                        ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_MODEL_BUILDDIR}
                        ${CMAKE_CURRENT_SOURCE_DIR}
                        ${CMAKE_CURRENT_SOURCE_DIR}/at_model
)

list(APPEND TARGET_CFLAGS   -O3)

math(EXPR FREQ_CL ${CONFIG_FREQUENCY_CLUSTER}/1000000 OUTPUT_FORMAT DECIMAL)
math(EXPR FREQ_FC ${CONFIG_FREQUENCY_FC}/1000000 OUTPUT_FORMAT DECIMAL)
math(EXPR FREQ_PE ${CONFIG_FREQUENCY_PERIPH}/1000000 OUTPUT_FORMAT DECIMAL)

message(STATUS "Clock speeds - Cluster ${FREQ_CL} FC ${FREQ_FC} Peripheral ${FREQ_PE}")
list(APPEND TARGET_PREPROCESSOR -DAT_MODEL_PREFIX=${MODEL_PREFIX}
                                -DSTACK_SIZE=${CONFIG_CL_MASTER_CORE_STACK_SIZE}
                                -DSLAVE_STACK_SIZE=${CONFIG_CL_SLAVE_CORE_STACK_SIZE}
                                -DFREQ_CL=${FREQ_CL}
                                -DFREQ_FC=${FREQ_FC}
                                -DFREQ_PE=${FREQ_PE}
)
LIST(APPEND TARGET_PREPROCESSOR -DPERF)




project(${TARGET_NAME} C ASM)

add_executable(${TARGET_NAME} ${TARGET_SRCS})
target_link_libraries(${TARGET_NAME} PUBLIC m)

target_compile_options(${TARGET_NAME} PUBLIC ${TARGET_CFLAGS} ${TARGET_PREPROCESSOR})
target_include_directories(${TARGET_NAME} PUBLIC ${TARGET_INCS})

###############################################################################
# CMake post initialization
###############################################################################
setupos(${TARGET_NAME})
target_print_header(${TARGET_NAME})
target_print_option(STATUS ${TARGET_NAME} ${Cyan} "Include directories" "${TARGET_INCS}" ${Cyan})
target_print_compile_options(STATUS ${TARGET_NAME} ${Cyan})
target_print_footer(${TARGET_NAME})
