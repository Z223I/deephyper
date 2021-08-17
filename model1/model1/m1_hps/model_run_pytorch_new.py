# Autogenerated by onnx-pytorch.

import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SAMBANOVA = False
DEVICE    = None
DTYPE     = None

if SAMBANOVA:
    from sambaflow import samba
    import sambaflow.samba.utils as utils
    from sambaflow.samba.utils.argparser import parse_app_args
    from sambaflow.samba.utils.pef_utils import get_pefmeta
    from model_args_pytorch import *

class Model(nn.Module):
    """Model class."""

    def __init__(self):
        """Initialize the model."""
        super(Model, self).__init__()
        self.__vars = nn.ParameterDict()
        for b in glob.glob(
            os.path.join(os.path.dirname(__file__), "variables", "*.npy")):
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
        return t_dense_6

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


@torch.no_grad()
def test_run_model(inputs=[torch.from_numpy(np.random.randn(*[1, 1690]).astype(np.float32))]):
    """Test the model."""
    model = Model()
    model.eval()
    print(model)
    rs = model(*inputs)
    print(rs[:10])
    return rs

def main_normal(argv):
    """Run main code."""
    config = {
        'proportion': .80,          # A value between [0., 1.] indicating how to split data between
                                    # training set and validation set. `prop` corresponds to the
                                    # ratio of data in training set. `1.-prop` corresponds to the
                                    # amount of data in validation set.
        'print_shape': 0            # Print the data shape.
    }

    from load_data_pytorch import load_data
    (x_train, y_train), (x_valid, y_valid) = load_data(config)

    arrayOf2dList = x_valid[0:1000]
    numpyArrayOf2dListFloat64 = np.array( arrayOf2dList )
    numpyArrayOf2dListFloat32 = numpyArrayOf2dListFloat64.astype(np.float32)
    torchTensorOf2dListFloat32 = torch.from_numpy( numpyArrayOf2dListFloat32 )
    inputs = [ torchTensorOf2dListFloat32 ]
    test_run_model( inputs )

def main_sn(argv):
    """Run main code."""
    utils.set_seed(256)
    args = parse_app_args(argv=argv, common_parser_fn=add_args, run_parser_fn=add_run_args)

    ipt, tgt = Model.get_fake_inputs(args)
    model = Model()
    model.summary()

    """
    model.eval()
    print(model)
    rs = model(*inputs)
    print(rs)
    """

    """
    [Info][SAMBA][Default] # Placing log files in pef/model_run_pytorch/model_run_pytorch.samba.log
    [Info][MAC][Default] # Placing log files in pef/model_run_pytorch/model_run_pytorch.mac.log
    /usr/local/lib/python3.7/site-packages/torch/nn/modules/container.py:569: UserWarning: Setting attributes on ParameterDict is not supported.
    warnings.warn("Setting attributes on ParameterDict is not supported.")
    """
    # This line probaly cannot take a compiled model.
    samba.from_torch_(model)

    inputs = (ipt, tgt)

if __name__ == '__main__':
    import sys

    if SAMBANOVA:
        main_sn(sys.argv[1:])
    else:
        main_normal(sys.argv[1:])