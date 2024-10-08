#if 0
#include "src/reduce_scatter/ths_op/gemm_reduce_scatter.h"

namespace phi {
namespace fusion {

#if 0
template<typename T, typename Context>
void GemmRSKernel(const Context& dev_ctx,
                  const DenseTensor& input,
                  const DenseTensor& weight,
                  paddle::optional<DenseTensor&> bias,
                  paddle::optional<DenseTensor&> input_scale,
                  paddle::optional<DenseTensor&> weight_scale,
                  paddle::optional<DenseTensor&> output_scale,
                  std::vector<DenseTensor>& output_buffers,
                  std::vector<DenseTensor>& reduce_buffers,
                  std::vector<DenseTensor>& barrier_buffers,
                  std::vector<DenseTensor>& sync_buffers,
                  const int32_t nnodes,
                  const int32_t max_m,
                  const int32_t n_dim,
                  bool transpose_weight,
                  bool fuse_reduction,
                  DenseTensor* fuck);
#endif

template<typename T, typename Context>
void GemmRSKernel(const Context& dev_ctx,
                  const DenseTensor& input,
                  const DenseTensor& weight,
                  paddle::optional<DenseTensor&> bias,
                  paddle::optional<DenseTensor&> input_scale,
                  paddle::optional<DenseTensor&> weight_scale,
                  paddle::optional<DenseTensor&> output_scale,
                  std::vector<DenseTensor>& output_buffers,
                  std::vector<DenseTensor>& reduce_buffers,
                  std::vector<DenseTensor>& barrier_buffers,
                  std::vector<DenseTensor>& sync_buffers,
                  const int32_t nnodes,
                  const int32_t max_m,
                  const int32_t n_dim,
                  bool transpose_weight,
                  bool fuse_reduction,
                  DenseTensor* fuck) {
  // maybe add a static cache in c++ side?
  phi::distributed::NCCLCommContext *comm_ctx = nullptr;
  comm_ctx = static_cast<phi::distributed::NCCLCommContext *>(dev_ctx.GetCommContext());
  int32_t rank = comm_ctx->GetRank();
  int32_t world_size = comm_ctx->GetSize();
  bytedance::flux::ths_op::flux::GemmRS<T, T> gemm_rs(dev_ctx,
                                                      nnodes,
                                                      max_m,
                                                      n_dim,
                                                      transpose_weight,
                                                      fuse_reduction,
                                                      rank,
                                                      world_size);
  gemm_rs.forward(input, weight, bias, input_scale, weight_scale, output_scale, true);
}
} // namespace fusion
} // namespace phi
#endif
