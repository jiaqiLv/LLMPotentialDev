"""
问题描述:
    对两个矩阵做乘法,将矩阵进行分块相乘运算,两种方式得到的结果无法通过equal测试
问题原因:
    Floating point operations are not associative. So the order in which you do the operation does matter.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == '__main__':
    batch_size = 10
    hidden_dim = 40
    output_dim = 20
    sliced_dim = hidden_dim//2

    dummy_mlp = nn.Linear(hidden_dim, output_dim, bias=False)
    parameters = torch.nn.Parameter(torch.randn(output_dim, hidden_dim))

    dummy_mlp.weight = parameters

    dummy_input = torch.randn(batch_size, hidden_dim)

    sliced_input_1, sliced_input_2 = torch.split(dummy_input, sliced_dim, dim=-1)

    sliced_output_1 = F.linear(sliced_input_1, dummy_mlp.weight[:, :sliced_dim])
    sliced_output_2 = F.linear(sliced_input_2, dummy_mlp.weight[:, sliced_dim:])

    final_output = dummy_mlp(dummy_input)

    print('=========TEST========')
    print('torch.allclose:', torch.allclose(final_output,sliced_output_1+sliced_output_2,rtol=1e-05,atol=1e-08,equal_nan =False))
    assert (final_output.shape == sliced_output_1.shape) and (sliced_output_1.shape == sliced_output_2.shape)
    diff = 0.0
    for i in range(final_output.shape[0]):
        for j in range(final_output.shape[1]):
            diff += torch.abs(final_output[i][j] - sliced_output_1[i][j] - sliced_output_2[i][j])
    print('diff:', diff)
    print('torch.equal:', torch.equal(final_output, sliced_output_1 + sliced_output_2))
    torch.testing.assert_close(final_output, sliced_output_1 + sliced_output_2, rtol=0.0, atol=0.0)
    print('=====================')

    # assert torch.equal(final_output, sliced_output_1 + sliced_output_2)
