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
  // paddle::distributed::ProcessGroupNCCL* pg = static_cast<paddle::distributed::ProcessGroupNCCL*>(map->get(ring_id));

  PADDLE_ENFORCE_NE(pg,
                    nullptr,
                    common::errors::Unavailable(
                        "ProcessGroup is nullptr."));

  printf("\npg->GetRank():%d, pg->GetSize():%d\n", pg->GetRank(), pg->GetSize());

  int32_t world_size = pg->GetSize();
  int32_t rank = pg->GetRank();

  // umiswing: idk why it's called comm_ctx, but it is the name in source code of process group.
  // const phi::GPUContext* comm_ctx = static_cast<phi::GPUContext*>(pg->GetDeviceContext(input.place(), /*use_calc_stream=*/false));
  const phi::GPUContext* comm_ctx = static_cast<phi::GPUContext*>(pg->GetGemmRSContext(input.place()));

  PADDLE_ENFORCE_NE(comm_ctx,
                    nullptr,
                    common::errors::Unavailable(
                        "comm_ctx is nullptr."));

  // const phi::DeviceContext* comm_ctx = pg->GetGemmRSContext(input.place());

#if 0
  const auto& comm_context_manager =
      phi::distributed::CommContextManager::GetInstance();

  PADDLE_ENFORCE_EQ(comm_context_manager.Has(std::to_string(ring_id)),
                    true,
                    common::errors::InvalidArgument(
                        "You choose to use new communication library by "
                        "setting environment "
                        "variable FLAGS_dynamic_static_unified_comm True. "
                        "But ring_id(%d) is "
                        "not found in comm_context_manager.",
                        std::to_string(ring_id)));
  phi::distributed::NCCLCommContext* comm_ctx = nullptr;
  comm_ctx = static_cast<phi::distributed::NCCLCommContext*>(
      comm_context_manager.Get(std::to_string(ring_id)));
  PADDLE_ENFORCE_NE(comm_ctx,
                    nullptr,
                    common::errors::Unavailable(
                        "NCCLCommContext is nullptr, collective op should "
                        "has ring_id attr."));
  PADDLE_ENFORCE_EQ(nranks,
                    comm_ctx->GetSize(),
                    common::errors::InvalidArgument(
                        "The number of ranks (%d) you set of must "
                        "be equal to comm_ctx->GetSize() (%d).",
                        nranks,
                        comm_ctx->GetSize()));

  printf("\ncomm_ctx->GetRank():%d, comm_ctx->GetSize():%d\n",comm_ctx->GetRank(), comm_ctx->GetSize());
#endif

#if 0
  // maybe add a static cache in c++ side?
  phi::distributed::NCCLCommContext *comm_ctx = nullptr;
  comm_ctx = static_cast<phi::distributed::NCCLCommContext *>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(comm_ctx,
                    nullptr,
                    common::errors::Unavailable(
                        "NCCLCommContext is nullptr, collective op should "
                        "has ring_id attr."));
  assert(comm_ctx != nullptr);
  printf("\n>>>>>>>>>>start get rank!\n");
  int32_t rank = comm_ctx->GetRank();
  printf("\n>>>>>>>>>>start get size!\n");
  int32_t world_size = comm_ctx->GetSize();
  printf("\n>>>>>>>>>>start init GemmRS!\n");
  printf("\ncomm_ctx is %p\n", (void*)comm_ctx);
  // printf("\n>>>>>>>>>>world_size is %d\n", world_size);
  // printf("\n>>>>>>>>>>rank is %d\n", rank);
#endif
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

  printf("\noutput_buffers:\n");
  for(auto p : output_buffers) {
    printf("%p, ",p);
  }
  printf("\n");
  for(auto p : nonconst_output_buffers) {
    printf("%p, ",p);
  }

  printf("\nreduce_buffers:\n");
  for(auto p : reduce_buffers) {
    printf("%p, ",p);
  }
  printf("\n");
  for(auto p : nonconst_reduce_buffers) {
    printf("%p, ",p);
  }

  printf("\nsync_buffers:\n");
  for(auto p : sync_buffers) {
    printf("%p, ",p);
  }
  printf("\n");
  for(auto p : nonconst_sync_buffers) {
    printf("%p, ",p);
  }

  printf("\nbarrier_buffers:\n");
  for(auto p : barrier_buffers) {
    printf("%p, ",p);
  }
  printf("\n");
  for(auto p : nonconst_barrier_buffers) {
    printf("%p, ",p);
  }

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

  printf("\n>>>>>>>>>>successful init GemmRS!\n");
  *fake_output = gemm_rs.forward(input, weight, bias, input_scale, weight_scale, output_scale, true);
  cudaDeviceSynchronize();
  printf("\n>>>>>>>>>>gemm_rs.forward finished\n");
}

} // namespace phi

PD_REGISTER_KERNEL(gemm_rs,
                   GPU,
                   ALL_LAYOUT,
                   phi::GemmRSKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
}
