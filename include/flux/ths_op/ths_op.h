//===- ths_op.h --------------------------------------------------- C++ ---===//
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

#pragma once
#include "flux/flux.h"
#include "flux/utils.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "./util.h"
#include "flux_shm.h"

#define FLUX_TORCH_EXTENSION_NAME flux_ths_pybind

namespace bytedance {
namespace flux {
namespace ths_op {

DataTypeEnum from_paddle_dtype(phi::DataType paddle_dtype);
bool is_fp8_paddle_dtype(phi::DataType paddle_dtype);
size_t paddle_dtype_size(phi::DataType paddle_dtype);

#if 0
// used by MoE
DenseTensor setup_shared_memory(
    int64_t rank, int64_t world_size, DenseTensor local_data, std::vector<void *> *host_ptrs);
#endif

#if 0
// umiswing: test_gemm_rs.py does not tune, it is weird but ... let us just ignore it temp.
// Wraps c++ types in class holder, in order to communicate with python
struct PyTuningRecord : public torch::CustomClassHolder {
  UnifiedGemmMeta meta;
  RuntimeConfig rt_conf;
  UnifiedGemmHParams best_hparams;

  PyTuningRecord(UnifiedGemmMeta, RuntimeConfig, UnifiedGemmHParams);
};

// add a tuning record to TuningConfigRegistry
void load_tuning_record(PyTuningRecord const &record);

class ProfilingContext : public torch::CustomClassHolder {
 private:
  TuningConfigGenerator codegen;

  using TopHParams = std::map<std::pair<float, int>, UnifiedGemmHParams>;
  std::map<std::pair<UnifiedGemmMeta, RuntimeConfig>, TopHParams> prof_results;

  int counter;
  std::unique_ptr<std::pair<UnifiedGemmMeta, RuntimeConfig>> latest_key_ptr;

  std::string to_string_topk(TopHParams const &top_hparams, int topk) const;

 public:
  static constexpr int kReturnTopK = 5;

  ProfilingContext(std::string name);

  TuningConfigGenerator const &get_codegen() const;

  // get generated code
  std::string get_code() const;

  // get all prof results as a vector, each element is the prof result of
  // a (GemmMeta,RuntimeConf) pair.
  std::vector<std::string> get_all_prof_results() const;

  std::vector<PyTuningRecord> get_all_records() const;

  // the prof result of the latest (GemmMeta, RuntimeConf) pair that has
  // finished profiling (i.e. record_best() has been called)
  std::string get_latest_prof_result() const;

  PyTuningRecord get_latest_record() const;

  // add a single record
  void add(
      UnifiedGemmMeta const &meta,
      RuntimeConfig const &rt_conf,
      UnifiedGemmHParams hparams,
      float elapsed_ms);

  // called after all records of (meta,rt_conf) have been added
  // this function will: 1. append the best config record of (meta,rt_conf) to codegen;
  // 2. update the latest_key_ptr to be (meta,rt_conf)
  UnifiedGemmHParams record_best(UnifiedGemmMeta const &meta, RuntimeConfig const &rt_conf);
};
#endif

#if 0
// umiswing: this class is for pybind in flux, we do not need it.
namespace py = pybind11;

// Registry of functions that register
// functions into module
class ThsOpsInitRegistry {
 public:
  using OpInitFunc = std::function<void(py::module &)>;
  static ThsOpsInitRegistry &instance();
  void register_one(std::string name, OpInitFunc &&func);
  void initialize_all(py::module &m) const;

 private:
  std::map<std::string, OpInitFunc> registry_;
  mutable std::mutex register_mutex_;

  ThsOpsInitRegistry() {}
  ThsOpsInitRegistry(const ThsOpsInitRegistry &) = delete;
  ThsOpsInitRegistry &operator=(const ThsOpsInitRegistry &) = delete;
};

struct DistEnvTP : public DistEnv {
  c10::intrusive_ptr<c10d::ProcessGroup> tp_group;
  DistEnvTP(c10::intrusive_ptr<c10d::ProcessGroup> tp_group, int nnodes = 1);
  std::string toString() const;
};

struct DistEnvTPWithEP : public DistEnv {
  c10::intrusive_ptr<c10d::ProcessGroup> tp_group;
  c10::intrusive_ptr<c10d::ProcessGroup> ep_group;
  int32_t ep_rank;
  int32_t ep_size;
  int32_t ffn_tp_size;
  int32_t ffn_tp_rank;

  DistEnvTPWithEP(
      c10::intrusive_ptr<c10d::ProcessGroup> tp_group,
      int nnodes = 1,
      c10::intrusive_ptr<c10d::ProcessGroup> ep_group = nullptr);
  std::string toString() const;
};

struct MoeArguments : public torch::CustomClassHolder {
  const int32_t max_ntokens;
  const int32_t hidden;
  const int32_t ffn_hidden;
  const int32_t nexperts;
  const int32_t topk;

  const c10::ScalarType input_dtype;
  const c10::ScalarType output_dtype;

  MoeArguments(
      int32_t max_ntokens,
      int32_t hidden,
      int32_t ffn_hidden,
      int32_t nexperts,
      int32_t topk,
      c10::ScalarType input_dtype,
      c10::ScalarType output_dtype);
};
#endif

#if 0
// umiswing: useless until ag_gemm
bool bitwise_check(DenseTensor A, DenseTensor B);
#endif

#if 0
void uniform_initialize(DenseTensor tensor, uint64_t seed, double min, double max);
void cudaipc_barrier_all_on_stream(
    cudaStream_t stream, std::vector<DenseTensor> &sync_buffer, int rank);
void lazy_init_buffer_tensor(DenseTensor *tensor, int64_t buffer_size);
#endif
}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
