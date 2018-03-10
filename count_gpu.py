import tensorflow as tf
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

config = tf.ConfigProto(
    # device_count={"GPU":2}, # GPUの数0に
    gpu_options=tf.GPUOptions(
        visible_device_list="0,1",
        per_process_gpu_memory_fraction=0.5  # 最大値の50%まで
    ),
    log_device_placement=True,
)

with tf.device("/gpu:0"):
  # GPUを使うように
  x = tf.random_uniform(shape=(10000, 10000))
  y = tf.random_uniform(shape=(10000, 10000))

with tf.device("/gpu:1"):
  # GPUを使うように
  z = tf.matmul(x, y)
  norm = tf.linalg.norm(z)

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
print(sess.run(norm))
