// #include "paddle/phi/kernels/fusion/flux/gemm_rs_kernel.h"
#include "src/reduce_scatter/ths_op/gemm_reduce_scatter.h"

namespace phi {

template<typename T, typename Context>
void GemmRSKernel(const Context& dev_ctx,
                  const DenseTensor& input,
                  const DenseTensor& weight,
                  const paddle::optional<DenseTensor>& bias,
                  const paddle::optional<DenseTensor>& input_scale,
                  const paddle::optional<DenseTensor>& weight_scale,
                  const paddle::optional<DenseTensor>& output_scale,
                  const std::vector<const DenseTensor*>& output_buffers,
                  const std::vector<const DenseTensor*>& reduce_buffers,
                  const std::vector<const DenseTensor*>& barrier_buffers,
                  const std::vector<const DenseTensor*>& sync_buffers,
                  const int32_t nnodes,
                  const int32_t max_m,
                  const int32_t n_dim,
                  bool transpose_weight,
                  bool fuse_reduction,
                  DenseTensor* fake_output) {
  // maybe add a static cache in c++ side?
  phi::distributed::NCCLCommContext *comm_ctx = nullptr;
  comm_ctx = static_cast<phi::distributed::NCCLCommContext *>(dev_ctx.GetCommContext());
  assert(comm_ctx != nullptr);
  printf("\n>>>>>>>>>>start get rank!\n");
  int32_t rank = comm_ctx->GetRank();
  int32_t world_size = comm_ctx->GetSize();
  printf("\n>>>>>>>>>>start init GemmRS!\n");
  bytedance::flux::ths_op::flux::GemmRS<T, T> gemm_rs(dev_ctx,
                                                      nnodes,
                                                      max_m,
                                                      n_dim,
                                                      transpose_weight,
                                                      fuse_reduction,
                                                      rank,
                                                      world_size);
  printf("\n>>>>>>>>>>successful init GemmRS!\n");
  gemm_rs.forward(input, weight, bias, input_scale, weight_scale, output_scale, true);
}

} // namespace phi

PD_REGISTER_KERNEL(gemm_rs,
                   GPU,
                   ALL_LAYOUT,
                   phi::GemmRSKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
}
