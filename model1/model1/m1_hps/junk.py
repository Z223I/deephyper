import os
import shutil
import tensorflow as tf
from tensorflow import keras
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()


model = keras.models.load_model('model_keras')

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

from onnx_pytorch import code_gen
code_gen.gen("model_onnx/model.onnx", "model_new_pytorch", overwrite=True)