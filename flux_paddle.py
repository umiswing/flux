import paddle
import paddle.distributed as dist
from paddle import _C_ops
from paddle.distributed import collective

from paddle.distributed.fleet.base import topology as tp
import numpy as np

# TODO(umiswing): ipc tensor is a bad name
def get_ipc_tensors(x):
  metadata = x.get_tensor()._share_cuda()
  metadata_list = []
  dist.all_gather_object(metadata_list, metadata)
  ipc_tensor_list = []
  for i in range(dist.get_world_size()):
    if i == dist.get_rank():
      ipc_tensor = paddle.to_tensor(x.get_tensor())
    else:
      # TODO(umiswing): need better api to manage ipc tensor
      ipc_tensor = paddle.tensor.creation.to_remote_tensor(paddle.base.core.LoDTensor._new_shared_cuda(metadata_list[i]))
    ipc_tensor_list.append(ipc_tensor)
  return ipc_tensor_list

def init_buffers(max_m, n_dim, output_dtype):
  reduce_buffer = paddle.empty(shape=[max_m, n_dim], dtype=output_dtype).zero_()
  # output_buffer = paddle.empty(shape=[max_m*2, n_dim],dtype=output_dtype).zero_()
  # umiswing: output_buffer shape here is very hack
  output_buffer = paddle.empty(shape=[max_m, n_dim],dtype=output_dtype).zero_()
  # TODO(umiswing): really need to find a way to init barrier_buffer from cpp, now just hack
  barrier_buffer = paddle.empty(shape=[6291456], dtype='uint8').zero_()

  sync_buffer = paddle.empty(shape=[dist.get_world_size()], dtype='int32').zero_()

  reduce_buffers = get_ipc_tensors(reduce_buffer)
  output_buffers = get_ipc_tensors(output_buffer)
  barrier_buffers = get_ipc_tensors(barrier_buffer)
  sync_buffers = get_ipc_tensors(sync_buffer)
  # TODO(umiswing): do zero
  return reduce_buffers, output_buffers, barrier_buffers, sync_buffers

def test_gemm_rs():
  dist.init_parallel_env()
  strategy = paddle.distributed.fleet.DistributedStrategy()

  strategy.hybrid_configs = {
    "mp_degree": 8,
    "dp_degree": 1,
    "pp_degree": 1,
  }

  paddle.distributed.fleet.init(is_collective=True, strategy=strategy)

  hcg = paddle.distributed.fleet.get_hybrid_communicate_group()

  model_parallel_group = hcg.get_model_parallel_group()

  world_size =  dist.get_world_size()
  rank = dist.get_rank()

  global_M = 8192
  global_N = 12288
  # global_K = 9216
  global_K = 12288

  local_K = global_K // dist.get_world_size()
  transpose_weight = False
  fuse_reduction = False
  dtype = 'bfloat16'
  scale = 1e-2
  delta = 1e-7

  seed = 2024
  paddle.seed(seed+dist.get_rank())
  np.random.seed(seed+dist.get_rank())

  input = paddle.randn(shape=[global_M, local_K], dtype=dtype) * scale - delta
  weight = paddle.randn(shape=[global_N, local_K], dtype=dtype) * scale - delta

  input_list = get_ipc_tensors(input)
  weight_list = get_ipc_tensors(weight)

  # if dist.get_rank() == 0:
  if True:
      global_input = paddle.concat(input_list, axis=1)
      global_weight = paddle.concat(weight_list, axis=1)

  reduce_buffers, output_buffers, barrier_buffers, sync_buffers = init_buffers(global_M, global_N, dtype)
  # print(reduce_buffers)
  # sync_buffers = None
  bias = None
  input_scale = None
  weight_scale = None
  output_scale = None
  # TODO(umiswing): better way to get nnodes
  nnodes = 1
  
  ring_id = model_parallel_group.id

  root_id = 0

  nranks = model_parallel_group.nranks

  output = _C_ops.gemm_rs(input,
                          weight,
                          bias,
                          input_scale,
                          weight_scale,
                          output_scale,
                          output_buffers,
                          reduce_buffers,
                          barrier_buffers,
                          sync_buffers,
                          nnodes,
                          global_M,
                          global_N,
                          transpose_weight,
                          fuse_reduction,
                          ring_id,
                          root_id,
                          nranks)
  print('==== phi::SumKernel input shape:', output.shape)
  if True:
      output_list = get_ipc_tensors(output)
      global_output = paddle.concat(output_list,axis=1).squeeze()
      # global_output_flatten = paddle.flatten(global_output)
      # print('global_output_flatten max atol val in bf16:', str(global_output_flatten[59849271].item()))
      # exit(0)

      if dist.get_rank()==0:
          if dtype == 'bfloat16':
            global_input = paddle.cast(global_input,'float32')
            global_weight = paddle.cast(global_weight,'float32')
            global_output = paddle.cast(global_output,'float32')
          global_input_np = global_input.numpy()
          global_weight_np = global_weight.numpy()
          global_output_np = global_output.numpy()
          np.save("paddle_npy/rank" + str(dist.get_rank()) + "_global_input.npy", global_input_np)
          np.save("paddle_npy/rank" + str(dist.get_rank()) + "_global_weight.npy", global_weight_np)
          np.save("paddle_npy/rank" + str(dist.get_rank()) + "_global_output.npy", global_output_np)

  '''
  if dist.get_rank()==0:
      if dtype == 'bfloat16':
        input = paddle.cast(input,'float32')
        weight = paddle.cast(weight,'float32')
        output = paddle.cast(output,'float32')
      input_np = input.numpy()
      weight_np = weight.numpy()
      output_np = output.numpy()
      np.save("paddle_npy/rank" + str(dist.get_rank()) + "_input.npy", input_np)
      np.save("paddle_npy/rank" + str(dist.get_rank()) + "_weight.npy", weight_np)
      np.save("paddle_npy/rank" + str(dist.get_rank()) + "_output.npy", output_np)
  '''
  print(input)

if __name__ == "__main__":
  test_gemm_rs()
