from tensorflow.contrib.eager.python import tfe
import tensorflow as tf


def scatter_update(ref: tfe.Variable, indices, updates):
	_ref = tfe.Variable(tf.cast(ref, tf.int32), trainable=False, name='hoge')
	del ref
	_updates = tf.cast(updates, tf.int32)
	x = tf.scatter_update(_ref, indices, _updates)
	update = tf.cast(x, tf.bool)
	return update


def main(_):
	ref = tfe.Variable([False, False, False], trainable=False, name='hoge')
	indices = tf.range(3)
	updates = tf.constant([True, True, True])
	
	print(scatter_update(ref, indices, updates))


if __name__ == '__main__':
	tfe.enable_eager_execution()
	tf.app.run()