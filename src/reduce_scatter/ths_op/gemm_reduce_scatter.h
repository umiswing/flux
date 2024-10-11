#pragma once
//===- gemm_reduce_scatter.cc ------------------------------------- C++ ---===//
//
// Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/runtime_config.h"
#include "flux/ths_op/ths_op.h"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/op_registry.h"
#include "flux/ths_op/util.h"
#include "flux/args/reduce_scatter.h"
#include "flux/utils.h"
// #include "reduce_scatter/ths_op/helper_ops.h"
#include "reduce_scatter/reduce_scatter_barrier_struct.hpp"
// #include "flux/ths_op/topo_utils.h"
#ifdef FLUX_REDUCE_SCATTERT_WITH_NCCL
#include "nccl.h"
#endif

#ifdef FLUX_SHM_USE_NVSHMEM
#include "nvshmemx.h"
#endif

////////////////////////////////////////////////////////////
#include "paddle/phi/core/utils/intrusive_ptr.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/fluid/distributed/collective/process_group_nccl.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/slice.h"
// #include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/unsqueeze_kernel.h"
#include "paddle/phi/kernels/view_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
////////////////////////////////////////////////////////////
namespace bytedance::flux::ths_op {
namespace flux {
// using torch::Tensor;

template<typename InT, typename OutT>
class GemmRS {
 private:
  const phi::GPUContext& dev_ctx;
  // const phi::distributed::ProcessGroup* pg;
  // umiswing: weird name, but it's used in source code of process group.
  const phi::GPUContext* comm_ctx;
  // const phi::DeviceContext* comm_ctx;
 private:
  // TODO(umiswing): i find nobody use phi::intrusive_ptr...
  // phi::intrusive_ptr<paddle::distributed::ProcessGroupNCCL> tp_group; // umiswing: not necessary to pass tp_group
  const int32_t nnodes;
  const int32_t max_m;
  const int32_t n_dim;
  const phi::DataType input_dtype = std::is_same<InT, phi::dtype::float16>::value ?
                                    phi::DataType::FLOAT16 :
                                    phi::DataType::BFLOAT16;
  const phi::DataType output_dtype = std::is_same<OutT, phi::dtype::float16>::value ?
                                     phi::DataType::FLOAT16 :
                                     phi::DataType::BFLOAT16;
  const bool transpose_weight;
  const bool fuse_reduction;

 private:
  const int32_t rank;
  const int32_t world_size;
  const int32_t local_world_size;
  const int32_t local_rank;
  const int32_t node_idx;

 private:
  // Symmetrically distributed tensor
  std::vector<DenseTensor*> output_buffers; // OutT
  std::vector<DenseTensor*> reduce_buffers; // OutT
  std::vector<DenseTensor*> barrier_buffers; //UINT8 (c10::ScalarType::Byte)
#ifndef FLUX_SHM_USE_NVSHMEM
  // used for the cuda-ipc-barrier
  std::vector<DenseTensor*> sync_buffers; // int32_t (c10::ScalarType::Int)
#endif
  DenseTensor* output_buffer;
  DenseTensor* reduce_buffer;
  DenseTensor* barrier_buffer{nullptr};
  DenseTensor gemm_buffer; // TODO(umiswing): find a way to unify declaration, phi has some requirements.
  std::vector<void *> output_scatter_ptrs;
  std::vector<void *> barrier_ptrs;
  bool no_nvlink;
  int sub_world_size;
  // phi::CUDAStream rs_stream_;
  cudaStream_t rs_stream_;
  cudaEvent_t event_;
  bool use_1d_ring;
  bool use_p2p_read;
  const bool is_fp8_gemm{false};

#ifdef FLUX_REDUCE_SCATTERT_WITH_NCCL
  ncclComm_t nccl_comm;
#endif
  void
  init_output_buffer() {
    // update max_m and allocate buffer

    if (get_arch() == _Sm90{} || no_nvlink || (get_arch() == _Sm80{} && nnodes > 1)) {
      // this->reduce_buffer = const_cast<DenseTensor*>(this->reduce_buffers[this->local_rank]);
      this->reduce_buffer = this->reduce_buffers[this->local_rank];
    } else {
      this->reduce_buffer = nullptr;
    }

    // this->output_buffer = const_cast<DenseTensor*>(this->output_buffers[this->local_rank]);
    this->output_buffer = this->output_buffers[this->local_rank];

    for (int i = 0; i < world_size; ++i) {
      if (i / this->local_world_size == rank / this->local_world_size) {
        // output_scatter_ptrs[i] = const_cast<void*>(this->output_buffers[i % this->local_world_size]->data());
        output_scatter_ptrs[i] = this->output_buffers[i % this->local_world_size]->data();
        // only check for ranks on the same node
        PADDLE_ENFORCE(
            output_scatter_ptrs[i] != nullptr, "nullptr buffr of rank %d", i);
      } else {
        output_scatter_ptrs[i] = nullptr;
      }
    }
#ifndef FLUX_SHM_USE_NVSHMEM
#if 0
    this->sync_buffers =
        flux_create_tensor_list({this->world_size}, paddle::DataType::INT32, this->tp_group);
#endif
#if 0
    // TODO(umiswing): maybe it's better to zero sync_buffers at python side
    // TODO(umiswing): what is type of sync_buffers? is it really OutT?
    phi::funcs::SetConstant<GPUContext, int32_t> set_functor;
    set_functor(this->dev_ctx, this->sync_buffers[this->rank], 0);
    // this->sync_buffers[this->rank].zero_();  // zeros the sync buffer for cuda ipc at the start
#endif
#endif
  }

  void
  lazy_init_barrier_buffer(int64_t buffer_size) {
    if ((buffer_size == 0) ||
        (barrier_buffer != nullptr && barrier_buffer->initialized() && buffer_size <= barrier_buffer->numel())) {
      return;
    }
#if 0
    this->barrier_buffers =
        flux_create_tensor_list({buffer_size}, paddle::DataType::UINT8, this->tp_group);
#endif
    this->barrier_buffer = this->barrier_buffers[this->local_rank];
    for (int i = 0; i < world_size; ++i) {
      if (i / this->local_world_size == rank / this->local_world_size) {
        barrier_ptrs[i] = this->barrier_buffers[i % this->local_world_size]->data();
        // only check for ranks on the same node
        // PADDLE_ENFORCE(barrier_ptrs[i] != nullptr, "nullptr buffr of rank " + std::to_string(i));
        PADDLE_ENFORCE_NOT_NULL(
            barrier_ptrs[i],
            common::errors::InvalidArgument("nullptr buffr of rank " + std::to_string(i)));
      } else {
        barrier_ptrs[i] = nullptr;
      }
    }
  }

  bool
  has_nvlink() {
    return true;
  }

  bool
  use_1d_ring_or_not() {
    ensure_nvml_init();
    // int devid = at::cuda::current_device();
    int devid = phi::backends::gpu::GetCurrentDeviceId();
    std::string devname(get_gpu_device_name(devid));
    if (devname != "NVIDIA L20" && world_size == 8) {
      return false;
    }
    return true;
  }

  bool
  use_p2p_read_or_not() {
    ensure_nvml_init();
    // int devid = at::cuda::current_device();
    int devid = phi::backends::gpu::GetCurrentDeviceId();
    std::string devname(get_gpu_device_name(devid));
    if (devname != "NVIDIA L20") {
      return true;
    }
    return false;
  }

  void
  lazy_init_gemm_buffer(DenseTensor input, int64_t buffer_size) {
    if (buffer_size <= 0) {
      return;
    }
    buffer_size = (buffer_size + 127) / 128 * 128;
    if (!this->gemm_buffer.initialized() || buffer_size > this->gemm_buffer.numel()) {
      // auto options = input.options().dtype(c10::ScalarType::Byte);
      // auto options = input.options().dtype(paddle::DataType::UINT8);
      // this->gemm_buffer = paddle::empty({buffer_size}, options);

      // TODO(umiswing): how to pass dev_ctx? what about options?
      // umiswing: paddle will pass dev_ctx when register kernel.
      this->gemm_buffer = phi::Empty<uint8_t>(this->dev_ctx,{buffer_size});
    }
  }

  // phi::CUDAStream
  // umiswing: paddle can only return comm stream of type cudaStram_t.
  cudaStream_t
  CreateReduceScatterStream() {
    return this->comm_ctx->stream();
#if 0
    // umiswing: i don't think it's a good idea to manually create and set cuda stream, and i don't understand why does
    // flux manage it in such way.
   phi::distributed::NCCLCommContext *comm_ctx = nullptr;
   comm_ctx = static_cast<phi::distributed::NCCLCommContext *>(
       dev_ctx.GetCommContext());

   // umiswing: paddle use different comm and calc stream by default, however, these two streams have the same
   // priority and paddle doesn't provide api to set stream priority.
   // TODO(umiswing): set comm stream to highest priority.
   return comm_ctx->GetStream();
#endif

#if 0
    // at::cuda::CUDAGuard guard(at::cuda::current_device());
    platform::CUDADeviceGuard guard(phi::backends::gpu::GetCurrentDeviceId());
    cudaStream_t rs_stream = nullptr;
    int least_priority, greatest_priority;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreateWithPriority(&rs_stream, cudaStreamNonBlocking, greatest_priority));
    // TODO(umiswing): it seems that paddle doesn't provide a synonymous api.
    return at::cuda::getStreamFromExternal(rs_stream, at::cuda::current_device());
#endif
  }

 public:
  GemmRS(
      const phi::GPUContext& dev_ctx,
      // const phi::distributed::ProcessGroup* pg,
      const phi::GPUContext* comm_ctx,
      // const phi::DeviceContext* comm_ctx,
      // phi::intrusive_ptr<paddle::distributed::ProcessGroupNCCL> tp_group_,
      int32_t nnodes,
      int32_t max_m,
      int32_t n_dim,
      bool transpose_weight,
      bool fuse_reduction,
      int32_t rank,
      int32_t world_size,
      std::vector<DenseTensor*>& output_buffers,
      std::vector<DenseTensor*>& reduce_buffers,
      std::vector<DenseTensor*>& sync_buffers,
      std::vector<DenseTensor*>& barrier_buffers)
      : dev_ctx(dev_ctx),
        comm_ctx(comm_ctx),
        // pg(pg),
        // tp_group(tp_group_),
        nnodes(nnodes),
        max_m(max_m),
        n_dim(n_dim),
#if 0
        input_dtype(input_dtype),
        output_dtype(output_dtype),
#endif
        transpose_weight(transpose_weight),
        fuse_reduction(fuse_reduction),
        rank(rank),
        world_size(world_size),
        output_buffers(output_buffers),
        reduce_buffers(reduce_buffers),
        sync_buffers(sync_buffers),
        barrier_buffers(barrier_buffers),
        local_world_size(world_size / nnodes),
        local_rank(rank % local_world_size),
        node_idx(rank / local_world_size),
        output_scatter_ptrs(world_size, nullptr),
        barrier_ptrs(world_size, nullptr),
        no_nvlink(!has_nvlink()),
        rs_stream_(CreateReduceScatterStream()),  // private stream. never dup with gemm stream
        use_1d_ring(use_1d_ring_or_not()),
        use_p2p_read(use_p2p_read_or_not()) {
#if 0
        is_fp8_gemm(is_fp8_paddle_dtype(input_dtype))
    if (std::is_same<InT, phi::dtype::float16>::value) {
        this->input_dtype = phi::DataType::FLOAT16;
    } else if (std::is_same<InT, phi::dtype::bfloat16>::value) {
        this->input_dtype = phi::DataType::BFLOAT16;
    }

    if (std::is_same<OutT, phi::dtype::float16>::value) {
        this->output_dtype = phi::DataType::FLOAT16;
    } else if (std::is_same<OutT, phi::dtype::bfloat16>::value) {
        this->output_dtype = phi::DataType::BFLOAT16;
    }

    this->is_fp8_gemm = is_fp8_paddle_dtype(this->input_dtype);
#endif

    PADDLE_ENFORCE(
        rank >= 0 && rank < world_size,
        "invalid rank: %d and world_size: %d",
        rank,
        world_size);
    PADDLE_ENFORCE(
        world_size % nnodes == 0,
        "invalid nnodes: world_size[%d] % nnodes[%d] !=0",
        world_size,
        nnodes);
    PADDLE_ENFORCE(
        !fuse_reduction || this->input_dtype == phi::DataType::FLOAT16,
        "Fuse reduction only support float16 type on SM80 due to instruction limitation.");
    this->init_output_buffer();
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreate(&event_));
// TODO(umiswing): adapt debug code, now just skip it.
#if defined(FLUX_DEBUG)
    if (no_nvlink) {
      LOG(WARNING) << "NvLink is not supported, seems running on a PCI-e machine.";
      ensure_nvml_init();
      int devid = at::cuda::current_device();
      std::string devname(get_gpu_device_name(devid));
      if (devname != "NVIDIA A100 80GB PCIe" && devname != "NVIDIA A800 80GB PCIe") {
        LOG(WARNING) << "Only NVIDIA A100/A800 80GB PCIe is tuned for. got " << devname;
      }
      if (world_size > 4 && world_size != 8) {
        LOG(WARNING) << "Only TensorParallel = 4 or 8 is tuned for. got " << world_size;
      }
      unsigned int gen = get_pcie_gen(devid);
      if (gen != 4) {
        LOG(WARNING) << "only PCI-e 4 version is tuned for. got PCI-e " << gen;
      }
    }
#endif
#ifdef FLUX_REDUCE_SCATTERT_WITH_NCCL
    if (nnodes > 1 && no_nvlink) {
      // TODO(umiswing): adapt this function, and codes in corresponding file.
      // nccl_comm = topo_utils::create_nccl_comm_with_processgroup(tp_group);
      // umiswing: idk why flux create nccl communicator manually.
      nccl_comm = this->dev_ctx.nccl_comm();
    } else {
      nccl_comm = nullptr;
    }
#endif
  }

  // umiswing: should i call such cuda/nccl api directly in paddle?
  ~GemmRS() {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(event_));
    // PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(rs_stream_));
#ifdef FLUX_REDUCE_SCATTERT_WITH_NCCL
    if (nccl_comm) {
      // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload(ncclCommDestroy(nccl_comm)));
    }
#endif
  }

  auto
  get_gemm_meta(bool has_bias, bool fast_accum = false) {
    ArchEnum arch = get_arch();
    auto gemm_layout = transpose_weight ? _RRR{}() : _RCR{}();
#if 0
    auto input_dtype = from_paddle_dtype(this->input_dtype);
    auto output_dtype = from_paddle_dtype(this->output_dtype);
#endif

    auto input_dtype = from_paddle_dtype(this->input_dtype);
    auto output_dtype = from_paddle_dtype(this->output_dtype);

    auto dt_conf = make_gemm_dtype_config(
        input_dtype, input_dtype, has_bias ? output_dtype : _Void{}(), output_dtype);

    fast_accum = fast_accum & dt_conf.is_input_fp8();
    bool is_gemm_v2 = ((int)arch < (int)_Sm90{}());
    auto meta = make_gemm_meta(
        dt_conf,
        arch,
        _ReduceScatter{},
        gemm_layout,
        is_gemm_v2 ? _GemmV2{}() : _GemmV3{}(),
        is_gemm_v2 ? UnifiedImplMeta(make_gemm_v2_meta(fast_accum))
                   : UnifiedImplMeta(make_gemm_v3_meta(fast_accum)),
        make_reduce_scatter_meta(
            this->fuse_reduction,
            nnodes > 1        ? _AcrossNode{}()
            : this->no_nvlink ? _IntraNodePcie{}()
                              : _IntraNode{}()));
    return meta;
  }

  RuntimeConfig
  get_rt_conf(DenseTensor input, DenseTensor weight, paddle::optional<DenseTensor> bias) {
    CHECK_INPUT(input, this->input_dtype);
    CHECK_INPUT(weight, this->input_dtype);
    PADDLE_ENFORCE(input.dims().size() == 2, "input rank is not 2");
    PADDLE_ENFORCE(weight.dims().size() == 2, "weight rank is not 2");
    int32_t m = input.dims()[0];
    int32_t k = input.dims()[1];
    int32_t n = transpose_weight ? weight.dims()[1] : weight.dims()[0];

    if (bias.is_initialized()) {
      CHECK_INPUT((*bias), this->output_dtype);
      PADDLE_ENFORCE(bias->dims().size() == 2, "bias rank is not 2");
      PADDLE_ENFORCE(
          m == bias->dims()[0],
          "bias dim0 != m: %d vs %d",
          bias->dims()[0],
          m);
      PADDLE_ENFORCE(
          n == bias->dims()[1],
          "bias dim1 != n: %d vs %d",
          bias->dims()[1],
          n);
    }

    // row major for streamk, todo: make weight layout an option
    int32_t wk = transpose_weight ? weight.dims()[0] : weight.dims()[1];
    FLUX_CHECK_LE(m, this->max_m) << "m-dim greater than maximum possible value";
    FLUX_CHECK_EQ(n, this->n_dim) << "n-dim != expected n_dim";
    FLUX_CHECK_EQ(wk, k) << "weight k-dim mismatch";
    return make_runtime_config(m, n, k, make_reduce_scatter_runtime_config(world_size, nnodes));
  }

  void
  forward_gemm_impl(
      const DenseTensor& input,
      const DenseTensor& weight,
      paddle::optional<DenseTensor> bias,
      paddle::optional<DenseTensor> input_scale,
      paddle::optional<DenseTensor> weight_scale,
      paddle::optional<DenseTensor> output_scale,
      bool fast_accum,
      paddle::optional<const UnifiedGemmHParams> hparams) {
    auto meta = get_gemm_meta(/*has_bias=*/bias.is_initialized(), fast_accum);
    auto rt_conf = get_rt_conf(input, weight, bias);
    // get cutlass op
    OpRegistry::OpPtr cutlass_op;
    if (hparams.is_initialized()) {
      cutlass_op = OpRegistry::instance().get_op(meta, *hparams);
    } else {
      cutlass_op = OpRegistry::instance().get_op(meta, rt_conf);
    }

    // TODO(houqi.1993) using args instead of envs
    static int num_blocks = get_int_from_env("FLUX_RS_BLOCKS", 12);
    static bool use_barrier_queue = get_bool_from_env("FLUX_RS_USE_BARRIER_QUEUE", false);
    static bool use_gemmk = get_bool_from_env("FLUX_RS_USE_GEMMK", no_nvlink);
    static bool use_cudaMemcpyAsync = get_bool_from_env("FLUX_RS_USE_CUDA_MEMCPY_ASYNC", false);
    static int n_split = get_int_from_env("FLUX_RS_N_SPLIT", 1);
    static bool per_tile_flags = get_bool_from_env("FLUX_RS_PER_TILE_FLAGS", no_nvlink);
    ReduceScatterArguments reduce_scatter_args{
        .reduce_scatter_num_blocks = num_blocks,
        .rs_stream = rs_stream_,
        .event = event_,
        .use_barrier_queue = use_barrier_queue,
        .use_gemmk = use_gemmk,
        .per_tile_flags = per_tile_flags,
        .use_cudaMemcpyAsync = use_cudaMemcpyAsync,
        .n_split = n_split,
        .sub_world_size = this->sub_world_size,
#ifdef FLUX_REDUCE_SCATTERT_WITH_NCCL
        .opaque = nccl_comm,
#else
        .opaque = nullptr,
#endif
        .use_1d_ring = use_1d_ring,
        .use_p2p_read = use_p2p_read,
    };
    // cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    cudaStream_t stream = this->dev_ctx.stream();

    if (!is_fp8_gemm) {
      FLUX_CHECK(!input_scale.is_initialized());
      FLUX_CHECK(!weight_scale.is_initialized());
      FLUX_CHECK(!output_scale.is_initialized());

      printf("\n>>>>>>>>>>>>>>>>>>>>>>>>> not fp8 gemm\n");


      const GemmReduceScatterArguments args{
          .m = rt_conf.m(),
          .n = rt_conf.n(),
          .k = rt_conf.k(),
          .rank = static_cast<int>(this->rank),
          .world_size = static_cast<int>(this->world_size),
          .nnodes = static_cast<int>(this->nnodes),
          .alpha = 1.0f,
          .beta = bias.is_initialized() ? 1.0f : 0.0f,
          .input = input.data(),
          .weight = weight.data(),
          .bias = bias.is_initialized() ? bias->data() : nullptr,
          .output_scatter_ptrs = this->output_scatter_ptrs.data(),
          .local_reduce_buffer =
              this->reduce_buffer != nullptr && this->reduce_buffer->initialized() ? this->reduce_buffer->data() : nullptr,
          .barrier_ptrs = this->barrier_ptrs.data(),
          .avail_sms = no_nvlink ? 1 : -1,
          .reduce_scatter_args = reduce_scatter_args
      };

      // initialize workspace
      int64_t workspace_size = cutlass_op->get_workspace_size(args);
      printf("\n>>>>>>>>>>workspace_size:%d\n<<<<<<<<<<",workspace_size);
      this->lazy_init_gemm_buffer(input, workspace_size);
      void *workspace = this->gemm_buffer.initialized() ? this->gemm_buffer.data() : nullptr;
      printf("\nafter workspace init\n");

      // initialize barrier workspace
      int64_t barrier_workspace_size = cutlass_op->get_barrier_workspace_size(args);
      // * 8 is for corner case reduce_scatter tiles. never mind this won't be a large memory
      barrier_workspace_size = barrier_workspace_size / sizeof(int) * sizeof(PerTileFlags) * 8;
      printf("\n>>>>>>>>>>barrier_workspace_size:%d\n<<<<<<<<<<",barrier_workspace_size);
      // TODO(umiswing): find a way to init barrier buffer in cpp, now just recored buffer size
      // and allocate in python manually
      this->lazy_init_barrier_buffer(barrier_workspace_size);
      printf("\nafter barrier workspace init\n");

      if ((fuse_reduction && !(meta.arch() == _Sm90{})) || this->no_nvlink) {
        // need to zero buffers;
        zero_buffers();
      }
      printf("\nafter zero buffer\n");
      cutlass_op->run(args, workspace, stream);

    } else {
      GemmReduceScatterFp8Arguments fp8_args{
          .m = rt_conf.m(),
          .n = rt_conf.n(),
          .k = rt_conf.k(),
          .rank = static_cast<int>(this->rank),
          .world_size = static_cast<int>(this->world_size),
          .nnodes = static_cast<int>(this->nnodes),
          .alpha = 1.0f,
          .beta = bias.is_initialized() ? 1.0f : 0.0f,
          .input = input.data(),
          .weight = weight.data(),
          .bias = bias.is_initialized() ? bias->data() : nullptr,
          .output_scatter_ptrs = this->output_scatter_ptrs.data(),
          .local_reduce_buffer =
              this->reduce_buffer->initialized() ? this->reduce_buffer->data() : nullptr,
          .barrier_ptrs = this->barrier_ptrs.data(),
          .avail_sms = no_nvlink ? 1 : -1,
          .reduce_scatter_args = reduce_scatter_args,
          .Aux = nullptr,
          .Vector = bias.is_initialized() ? bias->data() : nullptr,
          .abs_max_Aux = nullptr,
          .abs_max_D = nullptr,
          .scaleA = (float *)(input_scale.is_initialized() ? input_scale->data() : nullptr),
          .scaleB = (float *)(weight_scale.is_initialized() ? weight_scale->data() : nullptr),
          .scaleC = nullptr,
          .scaleD = (float *)(output_scale.is_initialized() ? output_scale->data() : nullptr),
          .scaleAux = nullptr};

      // initialize workspace
      int64_t workspace_size = cutlass_op->get_workspace_size(fp8_args);
      this->lazy_init_gemm_buffer(input, workspace_size);
      void *workspace = this->gemm_buffer.initialized() ? this->gemm_buffer.data() : nullptr;

      // initialize barrier workspace
      int64_t barrier_workspace_size = cutlass_op->get_barrier_workspace_size(fp8_args);
      // * 8 is for corner case reduce_scatter tiles. never mind this won't be a large memory
      barrier_workspace_size = barrier_workspace_size / sizeof(int) * sizeof(PerTileFlags) * 8;
      this->lazy_init_barrier_buffer(barrier_workspace_size);

      if ((fuse_reduction && !(meta.arch() == _Sm90{})) || this->no_nvlink) {
        // need to zero buffers;
        zero_buffers();
      }
      cutlass_op->run(fp8_args, workspace, stream);
    }
    cudaDeviceSynchronize();
    printf("\n>>>>>>>>>>>>>>forward_gemm_impl finished\n");
  }

  DenseTensor
  forward_reduce_scatter_impl(
      DenseTensor input,
      DenseTensor weight,
      paddle::optional<DenseTensor> bias,
      paddle::optional<const UnifiedGemmHParams> hparams) {
    auto meta = get_gemm_meta(/*has_bias=*/bias.is_initialized());  // fast_accum doesn't matter
    auto rt_conf = get_rt_conf(input, weight, bias);

    // get cutlass op
    OpRegistry::OpPtr cutlass_op;
    if (hparams.is_initialized()) {
      cutlass_op = OpRegistry::instance().get_op(meta, *hparams);
    } else {
      cutlass_op = OpRegistry::instance().get_op(meta, rt_conf);
    }

    int m = rt_conf.m();
    int n = rt_conf.n();

    if (((int)get_arch() < (int)_Sm90{}())) {
      // auto full_output = this->output_buffer.slice(0, 0, m);
      DenseTensor full_output = phi::funcs::Slice<OutT>(this->dev_ctx, *this->output_buffer, {0}, {0}, {m});
      if (nnodes > 1 && !no_nvlink) {
#if 0
        // TODO (umiswing): support multinodes reduce
        // printf("fuse_reduction:%d \n\n", fuse_reduction);
        auto unified_hparams = cutlass_op->get_runtime_gemm_hparams();
        auto tile_shape = unified_hparams.tile_shape();
        auto [tile_M, tile_N, tile_K] = tile_shape;
        int m_rank = m / world_size;
        // auto result = torch::empty({m_rank, n}, this->output_buffer.options());
        // TODO(umiswing): not sure how to set tensor options, just ignore it now.
        DenseTensor result = paddle::Empty<OutT>(this->dev_ctx, {m_rank, n});
#if 0
        auto output_to_reduce =
            this->reduce_buffer.slice(0, 0, nnodes * m_rank).view({nnodes, m_rank, this->n_dim});

        // TODO(umiswing): make sure paddle::Reshape does not deep copy the src tensor.
        DenseTensor output_to_reduce = paddle::Reshape<output_dtype>(this->dev_ctx,
                                                                     phi::funcs::Slice<output_dtype>(this->dev_ctx,
                                                                                                     this->reduce_buffer,
                                                                                                     {0},
                                                                                                     {0},
                                                                                                     {nnodes * m_rank}),
                                                                     {nnodes, m_rank, this->n_dim});
#endif
       DenseTensor output_to_reduce = phi::funcs::Slice<OutT>(this->dev_ctx,
                                                                      this->reduce_buffer,
                                                                      {0},
                                                                      {0},
                                                                      {nnodes * m_rank});
        phi::ViewShapeKernel(this->dev_ctx,
                             output_to_reduce,
                             {nnodes, m_rank, this->n_dim},
                             &output_to_reduce);


        bsr_reduce(output_to_reduce, result, tile_M, tile_N);
        return result;
        // return full_output;
#endif
      } else if (no_nvlink) {
        int m_per_rank = m / this->world_size;
#if 0
        auto output_2d =
            output_buffer.slice(0, m_per_rank * this->rank, m_per_rank * (this->rank + 1));
#endif
        DenseTensor output_2d = phi::funcs::Slice<OutT>(this->dev_ctx,
                                                        *this->output_buffer,
                                                        {0},
                                                        {m_per_rank * this->rank},
                                                        {m_per_rank * (this->rank + 1)});
        constexpr int kNumaWorldSize = 4;
        constexpr int kNumaNodes = 2;
        int local_world_size = world_size / nnodes;
        int local_rank = rank % local_world_size;
        int node_id = rank / local_world_size;
        int numa_id = local_rank / kNumaWorldSize;
        int rank_numa_local = local_rank % kNumaWorldSize;
        int rank_prev = (rank_numa_local - 1 + kNumaWorldSize) % kNumaWorldSize;
        rank_prev += numa_id * kNumaWorldSize + node_id * local_world_size;
        int rank_next = (rank_numa_local + 1) % kNumaWorldSize;
        rank_next += numa_id * kNumaWorldSize + node_id * local_world_size;
        int rank_from = numa_id == 0 ? rank_next : rank_prev;
        for (int i = 1; i < nnodes; i++) {
          int reduce_unused_segment = (rank_from + kNumaNodes + i * local_world_size) % world_size;
#if 0
          auto segment_other_node = reduce_buffer.slice(
              0, m_per_rank * reduce_unused_segment, m_per_rank * (reduce_unused_segment + 1));
#endif
          DenseTensor segment_other_node = phi::funcs::Slice<OutT>(this->dev_ctx,
                                                                   *reduce_buffer,
                                                                   {0},
                                                                   {m_per_rank * reduce_unused_segment},
                                                                   {m_per_rank * (reduce_unused_segment + 1)});
                                                                    
#if 0
          output_2d.add_(segment_other_node);
#endif
          phi::AddKernel<OutT, phi::GPUContext>(this->dev_ctx, output_2d, segment_other_node, &output_2d);
        }
        return output_2d;
      } else {
        int local_world_size = world_size / nnodes;
        if (fuse_reduction) {
          auto length = m / world_size;
          // TODO(umiswing): idk should i call inplace slice or not.
          // return this->output_buffer.slice(0, rank * length, (rank + 1) * length).unsqueeze(0);
          phi::DenseTensor output_buffer_sliced =
              phi::funcs::Slice<OutT>(this->dev_ctx,
                                              *this->output_buffer,
                                              {0},
                                              {0},
                                              {length});
          // auto axes = IntArray{0};
          phi::UnsqueezeKernel<OutT>(this->dev_ctx,
                               // const_cast<const DenseTensor&>(output_buffer_sliced),
                               output_buffer_sliced,
                               // output_buffer_sliced,
                               // const_cast<const IntArray&>(axes),
                               IntArray{0},
                               &output_buffer_sliced);
          return output_buffer_sliced;
                               
          // return this->output_buffer.slice(0, 0, length).unsqueeze(0);
        } else {
#if 0
          auto output_4d = full_output.view({nnodes, local_world_size, m / world_size, n});
          auto output = output_4d.sum(1);  // (nnodes,m_rank,n)
#endif
          DenseTensor output_4d;
          phi::ViewShapeKernel(this->dev_ctx,
                                      full_output,
                                      {nnodes, local_world_size, m / world_size, n},
                                      &output_4d);
          DenseTensor output;
          output.Resize(common::make_ddim({nnodes, m / world_size, n}));
          this->dev_ctx.template Alloc<OutT>(&output);
          phi::SumKernel<OutT>(this->dev_ctx, output_4d, {1}, this->output_dtype, false, &output);

          return output;
        }
      }
    } else if (meta.arch() == _Sm90{}) {
      int reduce_m_dim = m / world_size * nnodes * nnodes;
#if 0
      auto full_output = this->reduce_buffer.slice(0, 0, reduce_m_dim);
      auto output_4d = full_output.view({nnodes, nnodes, m / world_size, n});
#endif

      DenseTensor full_output = phi::funcs::Slice<OutT>(this->dev_ctx,
                                                        *this->reduce_buffer,
                                                        {0},
                                                        {0},
                                                        {reduce_m_dim});
      DenseTensor output_4d;
      phi::ViewShapeKernel(this->dev_ctx,
                           full_output,
                           {nnodes, nnodes, m / world_size, n},
                           &output_4d);
      if (nnodes == 1) {
#if 0
        auto output = output_4d[node_idx].sum(0);  // (m_rank,n)
#endif
        DenseTensor output;
        phi::ViewShapeKernel(this->dev_ctx,
                             output_4d,
                             {m / world_size, n},
                             &output);
        return output;

#if 0
        // TODO(umiswing): only 1 node, so just hack bcz idk how to do inplace scatter in paddle.
        DenseTensor output;
        output.Resize(common::make_dim(m / world_size, n));
        this->dev_ctx.template Alloc<OutT>(&output);
        phi::SumKernel<OutT>(this->dev_ctx,
                             output_4d[node_idx],
                             {0},
                             this->output_dtype,
                             false,
                             &output);
        return output;
#endif
      } else {
#if 0
        // TODO(umiswing): support multinodes reduce
        int m_rank = m / world_size;
        // auto output = torch::empty({m_rank, n}, output_buffer.options());
        DenseTensor output = paddle::Empty<OutT>(this->dev_ctx,
                                                 {m_rank, n});
        auto unified_hparams = cutlass_op->get_runtime_gemm_hparams();
        auto tile_shape = unified_hparams.tile_shape();
        auto [tile_M, tile_N, tile_K] = tile_shape;
        bsr_reduce(output_4d[node_idx], output, tile_M, tile_N);
        return output;
#endif
      }
    } else {
      PADDLE_ENFORCE(false, "unsupported arch:%s", std::string(enum_to_string(meta.arch())));
    }
  }

  DenseTensor
  forward_impl(
      const DenseTensor& input,
      const DenseTensor& weight,
      paddle::optional<DenseTensor> bias,
      paddle::optional<DenseTensor> input_scale,
      paddle::optional<DenseTensor> weight_scale,
      paddle::optional<DenseTensor> output_scale,
      bool fast_accum,
      paddle::optional<const UnifiedGemmHParams> hparams) {
    forward_gemm_impl(
        input, weight, bias, input_scale, weight_scale, output_scale, fast_accum, hparams);
    forward_barrier(input, weight, bias);
    return forward_reduce_scatter_impl(input, weight, bias, hparams);
  }

  void
  forward_gemm(
      const DenseTensor& input,
      const DenseTensor& weight,
      paddle::optional<DenseTensor> bias,
      paddle::optional<DenseTensor> input_scale,
      paddle::optional<DenseTensor> weight_scale,
      paddle::optional<DenseTensor> output_scale,
      bool fast_accum) {
    return forward_gemm_impl(
        input,
        weight,
        bias,
        input_scale,
        weight_scale,
        output_scale,
        fast_accum,
        // TODO(umiswing): should I use std::nullopt here?
        // c10::nullopt);
        paddle::none);
  }

  void
  forward_barrier(DenseTensor input, DenseTensor weight, paddle::optional<DenseTensor> bias) {
    // TODO(umiswing): not sure which paddle api to use.
    // cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    cudaStream_t stream = this->dev_ctx.stream();
    if (get_arch() == _Sm90{} and nnodes == 1) {
      // only local reduce, skip nvshmem barrier
    } else {
#ifdef FLUX_SHM_USE_NVSHMEM
      flux_barrier_all_on_stream(stream);
#else
      flux_barrier_all_on_stream(stream, this->sync_buffers, this->rank);
#endif
    }
  }

  DenseTensor
  forward_reduce_scatter(
      DenseTensor input, DenseTensor weight, paddle::optional<DenseTensor> bias) {
    return forward_reduce_scatter_impl(
        // TODO(umiswing): should I use std::nullopt here?
        input, weight, bias, paddle::none);
  }

  DenseTensor
  forward(
      const DenseTensor& input,
      const DenseTensor& weight,
      paddle::optional<DenseTensor> bias,
      paddle::optional<DenseTensor> input_scale,
      paddle::optional<DenseTensor> weight_scale,
      paddle::optional<DenseTensor> output_scale,
      bool fast_accum) {
    return forward_impl(
        input,
        weight,
        bias,
        input_scale,
        weight_scale,
        output_scale,
        fast_accum,
        // TODO(umiswing): should I use std::nullopt here?
        // c10::nullopt);
        paddle::none);
  }

  void
  zero_buffers() {
    // TODO(umiswing): not sure which api to use in paddle.
    // cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    cudaStream_t stream = this->dev_ctx.stream();
#if 0
    if (this->output_buffer.defined()) {
      this->output_buffer.zero_();
    }
    if (this->barrier_buffer.defined()) {
      this->barrier_buffer.zero_();
    }
    if (this->reduce_buffer.defined()) {
      this->reduce_buffer.zero_();
    }
#endif
    if (this->output_buffer->initialized()) {
        // FillKernel(this->dev_ctx, this->output_buffer, 0, &this->output_buffer);
        phi::funcs::SetConstant<GPUContext, OutT> set_functor;
        set_functor(this->dev_ctx, this->output_buffer, OutT{0});
    }
    if (this->barrier_buffer->initialized()) {
        // FillKernel(this->dev_ctx, this->barrier_buffer, 0, &this->barrier_buffer);
        phi::funcs::SetConstant<GPUContext, uint8_t> set_functor;
        set_functor(this->dev_ctx, this->barrier_buffer, uint8_t{0});
    }
    if (this->reduce_buffer->initialized()) {
        // FillKernel(this->dev_ctx, this->reduce_buffer, 0, &this->reduce_buffer);
        phi::funcs::SetConstant<GPUContext, OutT> set_functor;
        set_functor(this->dev_ctx, this->reduce_buffer, OutT{0});
    }
#ifdef FLUX_SHM_USE_NVSHMEM
    flux_barrier_all_on_stream(stream);
#else
    flux_barrier_all_on_stream(stream, this->sync_buffers, this->rank);
#endif
    if (!no_nvlink) {
      // TODO(umiswing): should i use cuda api or paddle api to sync stream?
      // c10::cuda::stream_synchronize(stream);
      phi::backends::gpu::GpuStreamSync(stream);
    }
  }

#if 0
  // TODO(umiswing): test_gemm_rs does not tune
  DenseTensor
  profiling(
      DenseTensor input,
      DenseTensor weight,
      paddle::optional<DenseTensor> bias,
      paddle::optional<DenseTensor> input_scale,
      paddle::optional<DenseTensor> weight_scale,
      paddle::optional<DenseTensor> output_scale,
      bool fast_accum,
      // TODO(umiswing): i find nobody use phi::intrusive_ptr...
      phi::intrusive_ptr<ProfilingContext> opt_ctx) {
#ifdef FLUX_SHM_USE_NVSHMEM
    auto meta = unify_type(this->get_gemm_meta(/*has_bias=*/bias.has_value(), fast_accum));
    auto rt_conf = this->get_rt_conf(input, weight, bias);
    ProfilingContext tmp_ctx("__tmp__");
    ProfilingContext *ctx = opt_ctx == nullptr ? &tmp_ctx : opt_ctx.get();

    auto filter_hparams = [&](UnifiedGemmHParams const &hparams) { return true; };

#if 0
    auto elapsed_tensor = paddle::empty({}, weight.options().dtype(paddle::DataType::FLOAT32));
    auto reduced_elapsed_tensor = elapsed_tensor.clone();
#endif
    DenseTensor elapsed_tensor = paddle::Empty<float>(this->dev_ctx, {});
    DenseTensor reduced_elapsed_tensor;
    reduced_elapsed_tensor.Resize(common::make_ddim({}));
    this->dev_ctx.template Alloc<float>(&reduced_elapsed_tensor);
    phi::Copy(this->dev_ctx, elapsed_tensor, this->dev_ctx.GetPlace(), false, &reduced_elapsed_tensor);

    OpRegistry::instance().visit_hparams(
        [&](UnifiedGemmHParams const &hparams) {
          if (not filter_hparams(hparams)) {
            return;
          }
          // filter non-consistent hparams
          constexpr int warm_iters = 5;
          constexpr int iters = 10;
          float total_elapsed = 0;

          // TODO(umiswing): not sure which api to use in paddle.
          // auto stream = c10::cuda::getCurrentCUDAStream();
          cudaStream_t stream = this->dev_ctx.stream();
          flux_barrier_all_on_stream(stream);
          // TODO(umiswing): should i use cuda api or paddle api to sync stream?
          // c10::cuda::stream_synchronize(stream);
          phi::backends::gpu::GpuStreamSync(stream);
          for (int iter = 0; iter < warm_iters + iters; ++iter) {
            GpuTimer timer;
            timer.start(stream);
            auto output [[maybe_unused]] = this->forward_impl(
                input, weight, bias, input_scale, weight_scale, output_scale, fast_accum, hparams);
            timer.stop();
            if (iter >= warm_iters) {
              total_elapsed += timer.elapsed_millis();
            }
          }

          // Avoid GPU frequency adjustment
          flux_barrier_all_on_stream(stream);
          // TODO(umiswing): should i use cuda api or paddle api to sync stream?
          // c10::cuda::stream_synchronize(stream);
          phi::backends::gpu::GpuStreamSync(stream);
          sleep(1);

          float avg_elapsed = int(total_elapsed / iters * 1000) / 1000.0;
#if 0
          // umiswing: why not fill?
          elapsed_tensor.copy_(torch::full({}, avg_elapsed));
#endif
          phi::funcs::SetConstant<GPUContext, float> set_functor;
          functor(this->dev_ctx, elapsed_tensor, avg_elapsed);

          nvshmemx_float_max_reduce_on_stream(
              NVSHMEM_TEAM_WORLD,
              static_cast<float *>(reduced_elapsed_tensor.data_ptr()),
              static_cast<float const *>(elapsed_tensor.data_ptr()),
              1,
              stream);

          float reduce_elapsed = reduced_elapsed_tensor.item().toFloat();
          ctx->add(meta, rt_conf, hparams, reduce_elapsed);
        },
        meta);

    auto best_hparams = ctx->record_best(meta, rt_conf);
    return this->forward_impl(
        std::move(input),
        std::move(weight),
        std::move(bias),
        std::move(input_scale),
        std::move(weight_scale),
        std::move(output_scale),
        fast_accum,
        std::move(best_hparams));
#else
    FLUX_CHECK(false) << "only support profiling when nvshmem is enabled";
    return torch::Tensor();
#endif
  }
#endif
};

}  // namespace flux

}  // namespace bytedance::flux::ths_op
