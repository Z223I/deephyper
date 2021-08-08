"""Model 1 model definition."""

from sambaflow import samba

import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Model class."""

    def __init__(self):
        """Initialize the model."""
        super(Model, self).__init__()
        self.__vars = nn.ParameterDict()
        for b in glob.glob(os.path.join(os.path.dirname(__file__), "variables", "*.npy")):
            v = torch.from_numpy(np.load(b))
            requires_grad = v.dtype.is_floating_point or v.dtype.is_complex
            self.__vars[os.path.basename(b)[:-4]] = nn.Parameter(
                torch.from_numpy(np.load(b)), requires_grad=requires_grad)


    def forward(self, *inputs):
        # sourcery skip: inline-immediately-returned-variable
        """Step forward in the model."""
        t_input_0, = inputs
        t_dense0 = torch.matmul(t_input_0, self.__vars["t_dense_kernel_0"])
        t_dense_20 = torch.matmul(t_input_0, self.__vars["t_dense_2_kernel_0"])
        t_biased_tensor_name3 = t_dense0 + self.__vars["t_dense_bias_0"]
        t_biased_tensor_name5 = t_dense_20 + self.__vars["t_dense_2_bias_0"]
        t_activation_Relu_0 = F.relu(t_biased_tensor_name3)
        t_dense_50 = torch.matmul(t_activation_Relu_0, self.__vars["t_dense_5_kernel_0"])
        t_dense_30 = torch.matmul(t_activation_Relu_0, self.__vars["t_dense_3_kernel_0"])
        t_dense_10 = torch.matmul(t_activation_Relu_0, self.__vars["t_dense_1_kernel_0"])
        t_biased_tensor_name1 = t_dense_50 + self.__vars["t_dense_5_bias_0"]
        t_biased_tensor_name4 = t_dense_30 + self.__vars["t_dense_3_bias_0"]
        t_biased_tensor_name6 = t_dense_10 + self.__vars["t_dense_1_bias_0"]
        t_activation_1_Relu_0 = F.relu(t_biased_tensor_name6)
        t_intermediate_tensor = t_activation_1_Relu_0 + t_biased_tensor_name5
        t_add_add_1_0 = t_intermediate_tensor + t_biased_tensor_name4
        t_activation_2_Relu_0 = F.relu(t_add_add_1_0)
        t_dense_40 = torch.matmul(t_activation_2_Relu_0, self.__vars["t_dense_4_kernel_0"])
        t_biased_tensor_name2 = t_dense_40 + self.__vars["t_dense_4_bias_0"]
        t_activation_3_Tanh_0 = torch.tanh(t_biased_tensor_name2)
        t_add_1_add_0 = t_activation_3_Tanh_0 + t_biased_tensor_name1
        t_activation_4_Relu_0 = F.relu(t_add_1_add_0)
        t_dense_60 = torch.matmul(t_activation_4_Relu_0, self.__vars["t_dense_6_kernel_0"])
        t_dense_6 = t_dense_60 + self.__vars["t_dense_6_bias_0"]
        #print(f"t_activation_4_Relu_0.size(): {t_activation_4_Relu_0.size()}")
        #print(f"t_dense_60.size(): {t_dense_60.size()}")
        #print(f"t_dense_6.size(): {t_dense_6.size()}")
        return t_dense_6

    @staticmethod
    def convert_data(inputs):
        """
        Convert data to correct format.

        Args:
            inputs: Input data.
            labels: Labels for inputs.
        """
        arrayOf2dList = inputs[0:10]
        numpyArrayOf2dListFloat64 = np.array( arrayOf2dList )
        numpyArrayOf2dListFloat32 = numpyArrayOf2dListFloat64.astype(np.float32)
        torchTensorOf2dListFloat32 = torch.from_numpy( numpyArrayOf2dListFloat32 )
        inputs = [ torchTensorOf2dListFloat32 ]

        return inputs

    @staticmethod
    def get_fake_inputs(args):
        """
        Get fake inputs.

        The size of the inputs are required for the SambaNova compiler.

        Args:
            args: CLI arguments.
        """
        inputs = samba.randn(args.batch_size, args.num_features, name='data', batch_dim=0).bfloat16().float()
        lablels = samba.randint(args.num_classes, (args.batch_size, ), name='label', batch_dim=0)

        inputs = Model.convert_data(inputs)

        return inputs, lablels


class FFN(nn.Module):
    """Feed Forward Network."""

    def __init__(self, num_features: int, ffn_dim_1: int, ffn_dim_2: int) -> None:
        """Initialize the class."""
        super().__init__()
        self.gemm1 = nn.Linear(num_features, ffn_dim_1, bias=False)
        self.relu = nn.ReLU()
        self.gemm2 = nn.Linear(ffn_dim_1, ffn_dim_2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Step forward."""
        out = self.gemm1(x)
        out = self.relu(out)
        out = self.gemm2(out)
        return out


class LogReg(nn.Module):
    """Logreg class."""

    def __init__(self, num_features: int, num_classes: int):
        """Initialize the class."""
        super().__init__()
        self.lin_layer = nn.Linear(in_features=num_features, out_features=num_classes, bias=False)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Step forward."""
        out = self.lin_layer(inputs)
        loss = self.criterion(out, targets)
        return loss, out


class FFNLogReg(nn.Module):
    """Feed Forward Network + LogReg."""

    def __init__(self, num_features: int, ffn_embedding_size: int, embedding_size: int, num_classes: int) -> None:
        """Initialize the class."""
        super().__init__()
        self.ffn = FFN(num_features, ffn_embedding_size, embedding_size)
        self.logreg = LogReg(embedding_size, num_classes)
        self._init_params()

    def _init_params(self) -> None:
        for p in self.parameters():
            nn.init.xavier_normal_(p)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Step forward."""
        out = self.ffn(inputs)
        loss, out = self.logreg(out, targets)
        return loss, out

    @staticmethod
    def get_fake_inputs(args):
        """
        Get fake inputs.

        The size of the inputs are required for the SambaNova compiler.

        Args:
            args: CLI arguments.
        """
        ipt = samba.randn(args.batch_size, args.num_features, name='image', batch_dim=0).bfloat16().float()
        tgt = samba.randint(args.num_classes, (args.batch_size, ), name='label', batch_dim=0)

        return ipt, tgt
