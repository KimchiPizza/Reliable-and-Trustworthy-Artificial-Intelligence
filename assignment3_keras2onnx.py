import tensorflow as tf
import tf2onnx

# Keras 모델 로드
model = tf.keras.models.load_model('mnist_cnn_trained.h5')

# ONNX 변환용 입력 스펙 정의
spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)

# Keras 모델을 ONNX로 변환 (opset 13)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path="mnist_cnn_trained.onnx")
