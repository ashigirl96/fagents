from tensorflow.contrib.eager.python import tfe
import tensorflow as tf
import numpy as np

tfe.enable_eager_execution()


class TwoLayerNetowrk(tfe.Network):
	
	def __init__(self, name):
		super(TwoLayerNetowrk, self).__init__(name=name)
		self.layer_one = self.track_layer(tf.layers.Dense(16, input_shape=(8,)))
		self.layer_one2 = self.track_layer(tf.layers.Dense(1000))
		self.layer_two = self.track_layer(tf.layers.Dense(1))
	
	def call(self, inputs):
		x = self.layer_one(inputs)
		x = self.layer_one2(x)
		x = self.layer_two(x)
		return x


def main(_):
	net = TwoLayerNetowrk('net')
	output = net(tf.ones([1, 8]))
	print([v.name for v in net.variables])


if __name__ == '__main__':
	tf.app.run()