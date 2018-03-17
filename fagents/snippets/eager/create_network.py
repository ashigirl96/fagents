from tensorflow.contrib.eager.python import tfe
import tensorflow as tf


class MNISTModel(tfe.Network):
	
	def __init__(self):
		super(MNISTModel, self).__init__()
		self.layer1 = self.track_layer(tf.layers.Dense(units=10))
		self.layer2 = self.track_layer(tf.layers.Dense(units=10))
	
	def call(self, input):
		result = self.layer1(input)
		result = self.layer2(result)
		return result


tfe.enable_eager_execution()


def foo():
	model = MNISTModel()
	batch = tf.zeros([2, 784])
	print(batch.shape)
	result = model(batch)
	print(result)


foo()