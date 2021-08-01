import os
import shutil
import tensorflow as tf
#from assets.tensorflow_to_onnx_example import create_and_train_mnist
from tensorflow import keras
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()


model = keras.models.load_model('model_keras')

#
# Valid activation functions.
# https://github.com/onnx/onnx/blob/master/docs/Operators.md#rnn
# Tanh should work.
#
# Current!! onnx-pytorch 0.1.3. pip install onnx-pytorch. Copy PIP instructions. Latest version.
# Released: May 13, 2021. Convert onnx to pytorch code.
#

"""
I met the same issue (TF 2.4.0, python 3.7.10, keras2onnx 1.7.0 ), my colleague solved the problem in this way:

add functions which disabled some actions in TF2
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
after adding the functions mentioned above, different error occur. Refer to this , revise the code
then the model can be converted successfully! Hope this help smile

https://github.com/onnx/keras-onnx/issues/659#issue-757119401
"""

# convert to onnx model
import keras2onnx
onnx_model = keras2onnx.convert_keras(model, model.name, target_opset=9)

#onnx_content = onnx_model.SerializeToString()
#print(f"onnx_content: {onnx_content}")

save_path = r"./model_onnx__nm"
if os.path.exists(save_path):
    shutil.rmtree(save_path)

import onnx
onnx.save(onnx_model, 'model_onnx/model.onnx')

# Or
"""
file = open("Sample_model.onnx", "wb")
file.write(onnx_model.SerializeToString())
file.close()
"""

"""
2021-07-30 07:57:04.209531: I tensorflow/core/platform/profile_utils/cpu_utils.cc:114] CPU Frequency: 1497600000 Hz
tf executing eager_mode: False
tf.keras model eager_mode: False
The ONNX operator number change on the optimization: 33 -> 22
The maximum opset needed by this model is only 8.
Traceback (most recent call last):
  File "tf_to_pytorch.py", line 47, in <module>
    code_gen.gen("model_onnx/model.onnx", "model_new_pytorch", overwrite=True, continue_on_error=True)
  File "/home/bwilson/venvonnx/lib/python3.8/site-packages/onnx_pytorch/code_gen.py", line 253, in gen
    assert os.path.exists(
AssertionError: model_new_pytorch is not empty and overwrite is not True.
(venvonnx) bwilson@bwilson-Inspiron-3593:~/DL/deephyper/model1/model1/m1_hp
"""
from onnx_pytorch import code_gen
code_gen.gen("model_onnx/model.onnx", "model_new_pytorch", overwrite=True)
