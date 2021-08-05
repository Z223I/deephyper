# Autogenerated by onnx-pytorch.

import glob
import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self._vars = nn.ParameterDict()
    self._regularizer_params = []
    for b in glob.glob(
        os.path.join(os.path.dirname(__file__), "variables", "*.npy")):
      v = torch.from_numpy(np.load(b))
      requires_grad = v.dtype.is_floating_point or v.dtype.is_complex
      self._vars[os.path.basename(b)[:-4]] = nn.Parameter(v, requires_grad=requires_grad)
    

  def forward(self, *inputs):
    input_0, = inputs
    dense0 = torch.matmul(input_0, self._vars["dense_kernel_0"])
    dense_20 = torch.matmul(input_0, self._vars["dense_2_kernel_0"])
    biased_tensor_name3 = torch.add(dense0, self._vars["dense_bias_0"])
    biased_tensor_name5 = torch.add(dense_20, self._vars["dense_2_bias_0"])
    activation_Relu_0 = F.relu(biased_tensor_name3)
    dense_50 = torch.matmul(activation_Relu_0, self._vars["dense_5_kernel_0"])
    dense_30 = torch.matmul(activation_Relu_0, self._vars["dense_3_kernel_0"])
    dense_10 = torch.matmul(activation_Relu_0, self._vars["dense_1_kernel_0"])
    biased_tensor_name1 = torch.add(dense_50, self._vars["dense_5_bias_0"])
    biased_tensor_name4 = torch.add(dense_30, self._vars["dense_3_bias_0"])
    biased_tensor_name6 = torch.add(dense_10, self._vars["dense_1_bias_0"])
    activation_1_Relu_0 = F.relu(biased_tensor_name6)
    intermediate_tensor = torch.add(activation_1_Relu_0, biased_tensor_name5)
    add_add_1_0 = torch.add(intermediate_tensor, biased_tensor_name4)
    activation_2_Relu_0 = F.relu(add_add_1_0)
    dense_40 = torch.matmul(activation_2_Relu_0, self._vars["dense_4_kernel_0"])
    biased_tensor_name2 = torch.add(dense_40, self._vars["dense_4_bias_0"])
    activation_3_Tanh_0 = torch.tanh(biased_tensor_name2)
    add_1_add_0 = torch.add(activation_3_Tanh_0, biased_tensor_name1)
    activation_4_Relu_0 = F.relu(add_1_add_0)
    dense_60 = torch.matmul(activation_4_Relu_0, self._vars["dense_6_kernel_0"])
    dense_6 = torch.add(dense_60, self._vars["dense_6_bias_0"])
    return dense_6

  
@torch.no_grad()
def test_run_model(inputs=[torch.from_numpy(np.random.randn(*[1, 1690]).astype(np.float32))]):
  model = Model()
  model.eval()
  rs = model(*inputs)
  print(rs)
  return rs
