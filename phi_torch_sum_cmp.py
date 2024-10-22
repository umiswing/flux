import paddle
import torch
import numpy as np

def np_assert_accuracy(np_a, np_b, rtol, atol):
    version_a = 'paddle.sum'
    version_b = 'torch.sum'
    max_atol_idx = np.argmax(np.abs(np_a - np_b))
    np_a_flatten = np_a.flatten()
    np_b_flatten = np_b.flatten()
    sub_res = np_a_flatten - np_b_flatten
    nonzero_idx = np.nonzero(np_b_flatten)
    sub_res = sub_res.take(nonzero_idx)
    np_b_flatten_nonzero = np_b_flatten.take(nonzero_idx).flatten()
    np_a_flatten_nonzero = np_a_flatten.take(nonzero_idx).flatten()
    if sub_res.size ==0:
        max_rtol_idx = 0 
    else:
        max_rtol_idx = np.argmax(np.abs(sub_res / np_b_flatten_nonzero))

    np.testing.assert_allclose(
        np_a,
        np_b,
        rtol=rtol,
        atol=atol,
        err_msg=(
            'max_atol value, {version_a}_value: {value_a}, {version_b}_value: {value_b},\n'.format(
                version_a=version_a,
                value_a=str(np_a_flatten[max_atol_idx].item()),
                version_b=version_b,
                value_b=str(np_b_flatten[max_atol_idx].item()),
            )   
            + 'max_rtol value , {version_a}_value: {value_a}, {version_b}_value: {value_b},\n'.format(
                version_a=version_a,
                value_a=str(np_a_flatten_nonzero[max_rtol_idx].item()) if max_rtol_idx < len(np_a_flatten_nonzero) else '', 
                version_b=version_b,
                value_b=str(np_b_flatten_nonzero[max_rtol_idx].item()) if max_rtol_idx < len(np_b_flatten_nonzero) else '', 
            )   
            + 'max_atol_idx: {max_atol_idx}\n'.format(
                max_atol_idx=max_atol_idx,
            )
        ),
    )
    print('pass')

shape = [1, 1024, 12288]
torch_shape = (1, 1024, 12288)

dtype = 'bfloat16'
torch_dtype = torch.bfloat16

scale = 1e-2
delta = 1e-7

seed = 123
for seed in range(2025, 2026):
    print(seed)
    paddle.seed(seed)
    np.random.seed(seed)
    input = paddle.randn(shape=shape, dtype=dtype) * scale - delta
    output = paddle.sum(input, axis=1)
    
    if dtype == "bfloat16":
      input = paddle.cast(input, 'float32')
      output = paddle.cast(output, 'float32')
    
    input_np = input.numpy()
    output_np = output.numpy()
 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch_input = torch.randn(torch_shape, dtype=torch_dtype, device="cuda") * scale - delta
    torch_output = torch.sum(torch_input, dim=1)

    if seed == 3:
      # 9423
      # print(output.flatten()[9423].item())
      # print(torch_output.flatten()[9423].item())
      print('{0:.60f}'.format(output.flatten()[9423].item()))
      print('{0:.60f}'.format(torch_output.flatten()[9423].item()))
      # print(format(output.flatten()[9423].item(), '.4g'))
      # print(format(torch_output.flatten()[9423].item(), '.3g'))
    
    if torch_dtype == torch.bfloat16:
      torch_input = torch_input.to(dtype=torch.float32)
      torch_output = torch_output.to(dtype=torch.float32)
    
    torch_input_np = torch_input.cpu().detach().numpy()
    torch_output_np = torch_output.cpu().detach().numpy()
    
    rtol = 0
    atol = 0

    np_assert_accuracy(input_np, torch_input_np, rtol, atol)
    np_assert_accuracy(output_np, torch_output_np, rtol, atol)
    del input
    del output
    del torch_input
    del torch_output
