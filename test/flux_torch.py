import argparse
from functools import partial
import os
import sys
import time
from typing import Union
import torch
import numpy as np
import datetime
import torch.distributed
from contextlib import nullcontext
import flux
# from flux.gemm_rs_sm80 import get_intra_node_pg_group

RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
NNODES = WORLD_SIZE // LOCAL_WORLD_SIZE

print(RANK, LOCAL_RANK, LOCAL_WORLD_SIZE, WORLD_SIZE, NNODES)

os.environ["NCCL_DEBUG"] = "ERROR"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_printoptions(precision=8)
# torch.manual_seed(3 + RANK)
# torch.cuda.manual_seed_all(3 + RANK)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
# np.random.seed(3 + RANK)

torch.distributed.init_process_group(
    backend="nccl", world_size=WORLD_SIZE, rank=RANK, timeout=datetime.timedelta(seconds=1800)
)
# use all ranks as tp group
TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
# print = partial(print, flush=True)

torch.cuda.set_device(LOCAL_RANK)
flux.init_flux_shm(TP_GROUP)

global_M = 8192
global_N = 12288
# global_K = 9216
global_K = 12288

local_K = global_K // TP_GROUP.size()
transpose_weight = False
fuse_reduction = False

dtype = torch.bfloat16
scale = 1e-2
delta = 1e-7

seed = 2024
torch.manual_seed(seed+RANK)
torch.cuda.manual_seed(seed+RANK)
# torch.cuda.manual_seed_all(seed+RANK)
np.random.seed(seed+RANK)

input = torch.randn((global_M, local_K), dtype=dtype, device="cuda") * scale - delta
weight = torch.randn((global_N, local_K), dtype=dtype, device="cuda") * scale - delta
bias = None
input_scale = None
weight_scale = None

cls = flux.GemmRS

gemm_rs_op = cls(
    TP_GROUP,
    NNODES,
    global_M,
    global_N,
    dtype,
    dtype,
    transpose_weight=transpose_weight,
    fuse_reduction=fuse_reduction)

output = gemm_rs_op.forward(
    input,
    weight,
    bias=bias,
    input_scale=input_scale,
    weight_scale=weight_scale,
    output_scale=None,
    fast_accum=False)

print('>>>>>> sum input shape:',output.shape)

output = torch.squeeze(output,0)

if RANK == 0:
  print(output)
  print(output.shape)

if dtype == torch.bfloat16:
  input = input.to(dtype=torch.float32)
  weight = weight.to(dtype=torch.float32)
  output = output.to(dtype=torch.float32)
np.save("torch_npy/rank" + str(RANK) + "_input.npy", input.cpu().detach().numpy())
np.save("torch_npy/rank" + str(RANK) + "_weight.npy", weight.cpu().detach().numpy())
np.save("torch_npy/rank" + str(RANK) + "_output.npy", output.cpu().detach().numpy())
