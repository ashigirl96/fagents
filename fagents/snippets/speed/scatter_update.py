import tensorflow as tf

import time

def main(_):
  # 3.735383319854736
  foo = tf.Variable(tf.random_uniform((1000, 3000), minval=0., maxval=1.), trainable=False)
  indices = tf.range(1000)
  bar = tf.random_uniform((3000, 3000), minval=0., maxval=1.)
  update = tf.scatter_update(foo, indices, foo @ bar)

  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    init.run()
    start_time = time.time()
    for _ in range(5):
      for i in range(1000):
        sess.run(update)
    process_time = time.time() - start_time
    print(process_time / 5)


if __name__ == '__main__':
  tf.app.run()