import numpy as np

def np_assert_accuracy(np_a, np_b, rtol, atol):
    version_a = 'paddle'
    version_b = 'torch'
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
            + 'max_atol_idx {max_atol_idx},\n'.format(
               max_atol_idx = max_atol_idx,
            )
        ),
    )

paddle_global_input = np.load("paddle_npy/rank0_global_input.npy")
paddle_global_weight = np.load("paddle_npy/rank0_global_weight.npy")
paddle_global_output = np.load("paddle_npy/rank0_global_output.npy")

torch_input_list = []
torch_weight_list = []
torch_output_list = []
for i in range(8):
  torch_input_list.append(np.load("torch_npy/rank" + str(i) + "_input.npy"))
  torch_weight_list.append(np.load("torch_npy/rank" + str(i) + "_weight.npy"))
  torch_output_list.append(np.load("torch_npy/rank" + str(i) + "_output.npy"))

torch_global_input = np.concatenate(torch_input_list, axis=1)
torch_global_weight = np.concatenate(torch_weight_list, axis=1)
torch_global_output = np.concatenate(torch_output_list, axis=0)

np_assert_accuracy(paddle_global_input, torch_global_input, rtol=0, atol=0)
np_assert_accuracy(paddle_global_weight, torch_global_weight, rtol=0, atol=0)
np_assert_accuracy(paddle_global_output, torch_global_output, rtol=0, atol=0)
np_assert_accuracy(paddle_global_output, torch_global_output, rtol=-1, atol=-1)
