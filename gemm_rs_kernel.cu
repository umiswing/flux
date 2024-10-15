// #include "paddle/phi/kernels/fusion/flux/gemm_rs_kernel.h"
#include "src/reduce_scatter/ths_op/gemm_reduce_scatter.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"

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
                  int ring_id,
                  int root_id,
                  int nranks,
                  DenseTensor* fake_output) {

  PADDLE_ENFORCE_GE(
      root_id,
      0,
      common::errors::InvalidArgument(
          "The root_id (%d) for c_scatter_op must be non-negative.", root_id));
  PADDLE_ENFORCE_GE(
      ring_id,
      0,
      common::errors::InvalidArgument(
          "The ring_id (%d) for c_scatter_op must be non-negative.", ring_id));

  auto map = distributed::ProcessGroupMapFromGid::getInstance();

  distributed::ProcessGroup* pg = map->get(ring_id);

  PADDLE_ENFORCE_NE(pg,
                    nullptr,
                    common::errors::Unavailable(
                        "ProcessGroup is nullptr."));

  int32_t world_size = pg->GetSize();
  int32_t rank = pg->GetRank();

  // umiswing: idk why it's called comm_ctx, but it is the name in source code of process group.
  // const phi::GPUContext* comm_ctx = static_cast<phi::GPUContext*>(pg->GetDeviceContext(input.place(), /*use_calc_stream=*/false));
  const phi::GPUContext* comm_ctx = static_cast<phi::GPUContext*>(pg->GetGemmRSContext(input.place()));

  PADDLE_ENFORCE_NE(comm_ctx,
                    nullptr,
                    common::errors::Unavailable(
                        "comm_ctx is nullptr."));

  auto get_non_const_buffers = [](const std::vector<const DenseTensor*>& buffers) {
    std::vector<DenseTensor*> nonconst_buffers;
#if 0
    nonconst_buffers.reserve(buffers.size());
    std::transform(buffers.begin(), buffers.end(), nonconst_buffers.begin(),
                   [] (const DenseTensor* p) { return const_cast<DenseTensor*>(p);});
#endif
    std::transform(buffers.begin(), buffers.end(), std::back_inserter(nonconst_buffers),
                   [] (const DenseTensor* p) { return const_cast<DenseTensor*>(p);});
    return nonconst_buffers;
  };
  std::vector<DenseTensor*> nonconst_output_buffers = get_non_const_buffers(output_buffers);
  std::vector<DenseTensor*> nonconst_reduce_buffers = get_non_const_buffers(reduce_buffers);
  std::vector<DenseTensor*> nonconst_sync_buffers = get_non_const_buffers(sync_buffers);
  std::vector<DenseTensor*> nonconst_barrier_buffers = get_non_const_buffers(barrier_buffers);

  bytedance::flux::ths_op::flux::GemmRS<T, T> gemm_rs(dev_ctx,
                                                      comm_ctx,
                                                      nnodes,
                                                      max_m,
                                                      n_dim,
                                                      transpose_weight,
                                                      fuse_reduction,
                                                      rank,
                                                      world_size,
                                                      nonconst_output_buffers,
                                                      nonconst_reduce_buffers,
                                                      nonconst_sync_buffers,
                                                      nonconst_barrier_buffers);

  *fake_output = gemm_rs.forward(input, weight, bias, input_scale, weight_scale, output_scale, true);
}

} // namespace phi

PD_REGISTER_KERNEL(gemm_rs,
                   GPU,
                   ALL_LAYOUT,
                   phi::GemmRSKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
}
