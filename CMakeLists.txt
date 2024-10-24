# set(CUTLASS_35_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cutlass)

# if(WITH_GPU)

set(GEMM_OP_REG_INP_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/gemm_op_registry.cu.in)
message("GEMM_OP_REG_INP_FILE ${GEMM_OP_REG_INP_FILE}")
# Generated op reg files and append to `GENERATED` file list                                                                            
function(flux_add_gemm_op_reg IMPL_HEADER IMPL_NAME NSPLITS GENERATED)                                                                  
  # set(GENERATED_REG_DIR "${CMAKE_CURRENT_BINARY_DIR}")                                                                                  
  set(GENERATED_REG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/generated/")                                                                                  
  # TODO(umiswing): auto detect arch
  set(CUDAARCHS "90" CACHE STRING "CUDA Architectures")
  message("GENERATED_REG_DIR ${GENERATED_REG_DIR}")
                                                                                                                                        
  set(GENERATED_FILES "${${GENERATED}}")                                                                                                
  math(EXPR max_value "${NSPLITS} - 1")                                                                                                 
  foreach(SPLIT_IDX RANGE 0 ${max_value})                                                                                               
    foreach(ARCH ${CUDAARCHS})                                                                                                          
      set(OUTPUT_FILE "${GENERATED_REG_DIR}/${IMPL_NAME}_sm${ARCH}_${SPLIT_IDX}_${NSPLITS}.cu")                                         
      configure_file(${GEMM_OP_REG_INP_FILE} ${OUTPUT_FILE} @ONLY)                                                                      
      list(APPEND GENERATED_FILES ${OUTPUT_FILE})                                                                                       
    endforeach()                                                                                                                        
  endforeach()                                                                                                                          
  set(${GENERATED} ${GENERATED_FILES} PARENT_SCOPE)                                                                                     
endfunction()
set(OP_REGS "")
flux_add_gemm_op_reg("reduce_scatter/gemm_v2_reduce_scatter.hpp" "GemmV2ReduceScatter" 16 "OP_REGS")
flux_add_gemm_op_reg("reduce_scatter/gemm_v3_reduce_scatter.hpp" "GemmV3ReduceScatter" 16 "OP_REGS")
file(
  GLOB FLUX_SRCS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "./*.cu"
  "./src/cuda/op_registry.cu"
  "./src/cuda/cuda_common.cc"
  "./src/ths_op/ths_op.cc"
  "./src/cuda/utils.cc"
  "./src/ths_op/flux_shm.cc"
  "./src/cuda/nvml_stub.cc"
  "./src/cuda/cudaipc_barrier_all.cu"
  "./src/cuda/op_registry.cu"
  "./src/cuda/op_registry_proto_utils.cc"
  "./src/reduce_scatter/tuning_config/*.cu"
  ${OP_REGS}
  # "gpu/flux/src/cuda/*.cu"
  # "gpu/flux/src/cuda/cuda_common.cc"
  # "gpu/flux/src/cuda/cuda_common.cc"
  )
list(REMOVE_ITEM FLUX_SRCS "./src/reduce_scatter/ths_op/gemm_reduce_scatter.cc" "./3rdparty/nccl/*")

set(FLUX_BUILD_TYPE
    SHARED
    CACHE INTERNAL "" FORCE)

set(FLUX_DEPS
    phi_gpu)

nv_library(
  flux ${FLUX_BUILD_TYPE}
  SRCS ${FLUX_SRCS}
  DEPS ${FLUX_DEPS})

#  21 include_directories("${CUTLASS_SOURCE_DIR}/")
#  22 include_directories("${CUTLASS_SOURCE_DIR}/include/")
#  23 include_directories("${CUTLASS_SOURCE_DIR}/tools/util/include/")

target_include_directories(flux PRIVATE "src"
                                        "include"
                                        "3rdparty/cutlass/include"
                                        "3rdparty/cutlass/tools/util/include")
                                        # ${MKL_INCLUDE})

find_library(CUDASTUB libcuda.so HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/
                                       REQUIRED)
target_link_libraries(flux
                      ${CUDASTUB})

get_property(FLUX_INCLUDE_DIRS TARGET flux PROPERTY INCLUDE_DIRECTORIES)
message("========================${FLUX_INCLUDE_DIRS}")
# string(REGEX REPLACE "(^|;)[^;]*third_party/cutlass[^;]*" ";;" NEW_FLUX_INCLUDE_DIRS ${FLUX_INCLUDE_DIRS})
# string(REGEX REPLACE "(^|;)[^;]*third_party/cutlass[^;]*($|;)" ";;" NEW_FLUX_INCLUDE_DIRS ${FLUX_INCLUDE_DIRS})
# string(REGEX REPLACE "(^|;)[^;]*third_party/cutlass[^;]*;?" "" NEW_FLUX_INCLUDE_DIRS ${FLUX_INCLUDE_DIRS})
# string(REGEX REPLACE "(^|;)[^;]+?third_party/cutlass[^;]+?;?" "" NEW_FLUX_INCLUDE_DIRS ${FLUX_INCLUDE_DIRS})
# string(REGEX REPLACE ";[^;]*third_party/cutlass[^;];" ";;" NEW_FLUX_INCLUDE_DIRS ${FLUX_INCLUDE_DIRS})

# 将包含路径字符串拆分为列表
string(REPLACE ";" ";" INCLUDE_DIRS_LIST "${FLUX_INCLUDE_DIRS}")

# 在列表中移除包含 third_party/cutlass 的路径
list(FILTER INCLUDE_DIRS_LIST EXCLUDE REGEX ".*third_party/cutlass.*")

# 将过滤后的列表重新组合为包含路径字符串
string(REPLACE ";" ";" NEW_FLUX_INCLUDE_DIRS "${INCLUDE_DIRS_LIST}")


message("========================${NEW_FLUX_INCLUDE_DIRS}")

set_property(TARGET flux PROPERTY INCLUDE_DIRECTORIES ${NEW_FLUX_INCLUDE_DIRS})
