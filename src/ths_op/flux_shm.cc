//===- flux_shm.cc ---------------------------------------------- C++ ---===//
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
#include <vector>
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/helper_kernels.h"
#include "flux/flux.h"
#include "flux/ths_op/flux_shm.h"

#ifdef FLUX_SHM_USE_NVSHMEM
#include <nvshmemx.h>
#endif

#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace bytedance {
namespace flux {

#ifdef FLUX_SHM_USE_NVSHMEM
namespace {
std::array<const char *, 5> kNvshmemInitStatus = {
    "NVSHMEM_STATUS_NOT_INITIALIZED",
    "NVSHMEM_STATUS_IS_BOOTSTRAPPED",
    "NVSHMEM_STATUS_IS_INITIALIZED",
    "NVSHMEM_STATUS_LIMITED_MPG",
    "NVSHMEM_STATUS_FULL_MPG"};
void
check_nvshmem_init() {
  FLUX_CHECK(nvshmemx_init_status() >= NVSHMEM_STATUS_IS_INITIALIZED)
      << "nvshmem not initialized: status " << kNvshmemInitStatus[nvshmemx_init_status()];
}
}  // namespace
torch::Tensor
nvshmem_create_tensor(const std::vector<int64_t> &shape, c10::ScalarType dtype) {
  check_nvshmem_init();
  auto option_gpu =
      at::TensorOptions().dtype(dtype).device(at::kCUDA).device_index(c10::cuda::current_device());
  auto size = torch::elementSize(dtype) *
              std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  FLUX_CHECK(size != 0);
  void *ptr = nvshmem_malloc(size);
  FLUX_CHECK(ptr != nullptr);
  CUDA_CHECK(cudaMemset(ptr, 0, size)); // memset the allocated buffer
  return at::from_blob(
      ptr, shape, [](void *ptr) { nvshmem_free(ptr); }, option_gpu);
}

std::vector<torch::Tensor>
nvshmem_create_tensor_list(const std::vector<int64_t> &shape, c10::ScalarType dtype) {
  check_nvshmem_init();
  auto option_gpu =
      at::TensorOptions().dtype(dtype).device(at::kCUDA).device_index(c10::cuda::current_device());
  auto size = torch::elementSize(dtype) *
              std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  FLUX_CHECK(size != 0);
  int local_world_size = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
  int rank = nvshmem_my_pe();
  int local_rank = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  std::vector<torch::Tensor> tensors;
  tensors.reserve(local_world_size);
  void *ptr = nvshmem_malloc(size);
  CUDA_CHECK(cudaMemset(ptr, 0, size)); // memset the allocated buffer
  FLUX_CHECK(ptr != nullptr);
  int rank_offset = rank - local_rank;
  for (int i = 0; i < local_world_size; i++) {
    // runs this call nvshmem failure, don't know why
    //  nvshmem_team_translate_pe(NVSHMEMX_TEAM_NODE, local_rank, NVSHMEM_TEAM_WORLD)
    int rank_global = i + rank_offset;
    if (rank == rank_global) {
      tensors.emplace_back(
          at::from_blob(ptr, shape, [](void *ptr) { nvshmem_free(ptr); }, option_gpu));
    } else {
      void *rptr = nvshmem_ptr(ptr, rank_global);
      FLUX_CHECK(rptr != nullptr) << "rank " << rank;
      tensors.emplace_back(at::from_blob(rptr, shape, option_gpu));
    }
  }

  return tensors;
}

#endif

// using Deleter = std::function<void(void*)>;
// using Deleter = std::function<void(phi::Allocation*)>;
using Deleter = void (*)(phi::Allocation*);
using AllocationDeleter = void (*)(phi::Allocation*);
DenseTensor from_blob(void *data,
                      const std::vector<int64_t>& shape,
                      phi::DataType dtype,
                      phi::Place place,
                      const Deleter& deleter,
                      phi::DataLayout layout = phi::DataLayout::NCHW ) {
  PADDLE_ENFORCE_NOT_NULL(
      data, common::errors::InvalidArgument("data can not be nullptr."));

  // TODO(umiswing): this check looks nice
  // auto data_place = GetPlaceFromPtr(data);
  phi::is_gpu_place(place);

  auto meta =
      phi::DenseTensorMeta(dtype, common::make_ddim(shape), layout);

  size_t size = SizeOf(dtype) * (meta.is_scalar ? 1 : product(meta.dims));

#if 0
  AllocationDeleter alloc_deleter = nullptr;
  if (deleter) {
    static thread_local Deleter g_deleter = deleter;
    alloc_deleter = [](phi::Allocation* p) { g_deleter(p); };
  }
#endif

  auto alloc =
      // std::make_shared<phi::Allocation>(data, size, alloc_deleter, place/*data_place*/);
      std::make_shared<phi::Allocation>(data, size, deleter, place/*data_place*/);

  return DenseTensor(alloc, meta);
}

std::vector<DenseTensor>
cudaipc_create_tensor_list(
    const std::vector<int64_t> &shape,
    const phi::DataType dtype,
    distributed::ProcessGroup* pg,
    const phi::GPUContext& dev_ctx,
    const std::string buffer_name,
    const bool real) {

  FLUX_CHECK(pg->GetSize() <= phi::backends::gpu::GetGPUDeviceCount())
      << "create_ipc_tensors should only be used intra node";

  size_t size = SizeOf(dtype) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  FLUX_CHECK(size != 0);
  void *ptr = nullptr;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&ptr, size));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemset(ptr, 0, size)); // memset the allocated buffer
  cudaIpcMemHandle_t handle;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaIpcGetMemHandle(&handle, ptr));

  DenseTensor handle_d = phi::Empty<uint8_t>(dev_ctx, {sizeof(cudaIpcMemHandle_t)});
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(
      handle_d.data(), &handle, sizeof(cudaIpcMemHandle_t), cudaMemcpyHostToDevice));
  long int handles_shape = sizeof(cudaIpcMemHandle_t) * pg->GetSize();
  DenseTensor handles_d = phi::Empty<uint8_t>(dev_ctx, {handles_shape});
  // TODO(umiswing): find a better way to wrap func params
  pg->AllGather(&handles_d, handle_d, 0, -1, true, true)->Wait();

  std::vector<cudaIpcMemHandle_t> handles_h(pg->GetSize());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(
      handles_h.data(),
      handles_d.data(),
      sizeof(cudaIpcMemHandle_t) * pg->GetSize(),
      cudaMemcpyDeviceToHost));

  std::vector<void *> ptrs(pg->GetSize());
  for (int i = 0; i < pg->GetSize(); ++i) {
    if (i != pg->GetRank()) {
      if (real) {
        PADDLE_ENFORCE_GPU_SUCCESS(cudaIpcOpenMemHandle(&ptrs[i], handles_h[i], cudaIpcMemLazyEnablePeerAccess));
      } else {
        void *ptr = nullptr;
        PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&ptr, size));
        PADDLE_ENFORCE_GPU_SUCCESS(cudaMemset(ptr, 0, size)); // memset the allocated buffer
        ptrs[i] = ptr;
      }
    } else {
      ptrs[i] = ptr;
    }
  }

  std::vector<DenseTensor> tensors;
  for (int i = 0; i < pg->GetSize(); ++i) {
    DenseTensor tensor;
    if (i == pg->GetRank() || !real) {
      tensor = from_blob(ptr, shape, dtype, dev_ctx.GetPlace(), [](phi::Allocation* allocation) { cudaFree(allocation->ptr()); });
    } else {
      tensor =
          from_blob(ptrs[i], shape, dtype, dev_ctx.GetPlace(), [](phi::Allocation* allocation) { cudaIpcCloseMemHandle(allocation->ptr()); });
    }
    tensors.emplace_back(tensor);
  }

  std::string filename = "debug.log" + std::to_string(pg->GetRank());

  static int clear_log UNUSED = [=]() {
    std::ofstream file(filename);
    file.close();
    return 0;
  }();

  std::ofstream file(filename, std::ios::app);
  file << "\n" << buffer_name;
  for (int i = 0; i < pg->GetSize(); ++i) {
    file << "i == pg->GetRank():" << (i == pg->GetRank()) << " ,i:" << i << " ,pg->GetRank():" << pg->GetRank();
  }
  file << std::endl;
  file.close();

  return tensors;
}

#if 0

void
init_flux_shm(c10::intrusive_ptr<c10d::ProcessGroup> c10_pg) {
#ifdef FLUX_SHM_USE_NVSHMEM
  nvshmemx_init_attr_t init_attr;
  init_attr.mpi_comm = (void *)c10_pg.get();  // bad! pretend I'm the MPIComm
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &init_attr);
  int mype = nvshmem_my_pe();
  CHECK(c10_pg->getRank() == mype)
      << "NVShmem init: rank does not match PE!" << c10_pg->getRank() << " vs " << mype;
#endif
}

torch::Tensor
flux_create_tensor(
    const std::vector<int64_t> &shape,
    c10::ScalarType dtype,
    c10::intrusive_ptr<c10d::ProcessGroup> pg) {
#ifdef FLUX_SHM_USE_NVSHMEM
  return nvshmem_create_tensor(shape, dtype);
#else
  FLUX_CHECK(false && "This line should never be reached");
  return torch::Tensor();
#endif
}
std::vector<torch::Tensor>
flux_create_tensor_list(
    const std::vector<int64_t> &shape,
    c10::ScalarType dtype,
    c10::intrusive_ptr<c10d::ProcessGroup> pg) {
#ifdef FLUX_SHM_USE_NVSHMEM
  return nvshmem_create_tensor_list(shape, dtype);
#else
  FLUX_CHECK(pg != nullptr);
  return cudaipc_create_tensor_list(pg, shape, dtype);
#endif
}
#endif

void
flux_barrier_all_on_stream(
    cudaStream_t stream,
    paddle::optional<std::vector<DenseTensor>> sync_buffers,
    paddle::optional<int> rank) {
#ifdef FLUX_SHM_USE_NVSHMEM
  nvshmemx_barrier_all_on_stream(stream);
#else
#if 0
  FLUX_CHECK(sync_buffers.has_value());
  FLUX_CHECK(rank.has_value());
#endif
  std::vector<int32_t *> sync_buffer_ptrs;
#if 0
  auto sync_buffers_val = sync_buffers.value();
  FLUX_CHECK(sync_buffers_val[rank.value()].defined());
#endif
  std::vector<DenseTensor>& sync_buffers_val = sync_buffers.get();
  FLUX_CHECK(sync_buffers_val[rank.get()].initialized());

  int world_size = sync_buffers_val.size();
  for (size_t i = 0; i < sync_buffers_val.size(); i++) {
    sync_buffer_ptrs.push_back(reinterpret_cast<int32_t *>(sync_buffers_val[i].data()));
  }
  cudaipc_barrier_all_on_stream_impl(stream, sync_buffer_ptrs.data(), rank.get(), world_size);
#endif
}

#if 0
void
pyflux_barrier_all_on_stream(
    intptr_t stream,
    c10::optional<std::vector<torch::Tensor>> sync_buffers,
    c10::optional<int> rank) {
  flux_barrier_all_on_stream((cudaStream_t)stream, sync_buffers, rank);
}
#endif

}  // namespace flux
}  // namespace bytedance
