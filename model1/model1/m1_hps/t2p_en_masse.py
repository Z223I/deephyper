import os
import shutil
import tensorflow as tf
#from assets.tensorflow_to_onnx_example import create_and_train_mnist
from tensorflow import keras
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

import keras2onnx
import onnx
from onnx_pytorch import code_gen

def convert_tf_to_pytorch(tf_dir):
    """Convert tensorflow model to PyTorch."""
    model = keras.models.load_model(tf_dir)

    # convert to onnx model
    onnx_model = keras2onnx.convert_keras(model, model.name, target_opset=9)

    #onnx_content = onnx_model.SerializeToString()
    #print(f"onnx_content: {onnx_content}")

    save_path_onnx = f"{tf_dir}_onnx__nm/model.onnx"
    if os.path.exists(save_path_onnx):
        shutil.rmtree(save_path_onnx)

    onnx.save(onnx_model, save_path_onnx)
    # Or
    """
    with open(save_path_onnx, "wb") as file:
        file.write(onnx_model.SerializeToString())
    """

    save_path_pytorch = f"{tf_dir}_pytorch"
    code_gen.gen(save_path_onnx, save_path_pytorch, overwrite=True)


for path in os.listdir('.'):
    if os.path.isdir(path):
        filename = os.path.basename(path)
        if 'model_' in filename:
            convert_tf_to_pytorch(filename)
