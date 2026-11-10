import tensorflow as tf

# TensorFlow가 GPU를 인식하는지 확인
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# GPU 사용을 강제로 테스트
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    c = tf.matmul(a, b)
    print(c)

