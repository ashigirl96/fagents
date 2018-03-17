"""Tests for the simulation operation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

from fagents import tools

from tensorflow.python import debug as tf_debug


class SimulateTest(tf.test.TestCase):
	def test_done_automatic(self):
		batch_env = self._create_test_batch_env((1, 2, 3, 4))
		algo = tools.MockAlgorithm(batch_env)
		done, _, _ = tools.simulate(batch_env, algo, log=False, reset=False)
		init = tf.global_variables_initializer()
		with self.test_session() as sess:
			sess.run(init)
			# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
			self.assertAllEqual([True, False, False, False], sess.run(done))
			self.assertAllEqual([True, True, False, False], sess.run(done))
			self.assertAllEqual([True, False, True, False], sess.run(done))
			self.assertAllEqual([True, True, False, True], sess.run(done))
	
	def test_done_forced(self):
		reset = tf.placeholder_with_default(False, ())
		batch_env = self._create_test_batch_env((2, 4))
		algo = tools.MockAlgorithm(batch_env)
		done, _, _ = tools.simulate(batch_env=batch_env,
																algo=algo,
																log=False,
																reset=reset)
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			self.assertAllEqual([False, False], sess.run(done))
			self.assertAllEqual([False, False], sess.run(done, feed_dict={reset: True}))
			self.assertAllEqual([True, False], sess.run(done))
			self.assertAllEqual([False, False], sess.run(done, feed_dict={reset: True}))
			self.assertAllEqual([True, False], sess.run(done))
			self.assertAllEqual([False, False], sess.run(done))
			self.assertAllEqual([True, True], sess.run(done))
	
	def test_reset_automatic(self):
		batch_env = self._create_test_batch_env((1, 2, 3, 4))
		algo = tools.MockAlgorithm(batch_env)
		done, _, _ = tools.simulate(batch_env, algo, log=False, reset=False)
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			for _ in range(10):
				sess.run(done)
		self.assertAllEqual([1] * 10, batch_env[0].steps)
		self.assertAllEqual([2] * 5, batch_env[1].steps)
		self.assertAllEqual([3, 3, 3, 1], batch_env[2].steps)
		self.assertAllEqual([4, 4, 2], batch_env[3].steps)
	
	def test_reset_forced(self):
		reset = tf.placeholder_with_default(False, ())
		batch_env = self._create_test_batch_env((2, 4))
		algo = tools.MockAlgorithm(batch_env)
		done, _, _ = tools.simulate(batch_env=batch_env,
																algo=algo,
																log=False,
																reset=reset)
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			sess.run(done)
			sess.run(done, {reset: True})
			sess.run(done)
			sess.run(done, {reset: True})
			sess.run(done)
			sess.run(done)
			sess.run(done)
		self.assertAllEqual([1, 2, 2, 2], batch_env[0].steps)
		self.assertAllEqual([1, 2, 4], batch_env[1].steps)
	
	def _create_test_batch_env(self, durations):
		envs = []
		for duration in durations:
			env = tools.MockEnvironment(
				observ_shape=(2, 3), action_shape=(3,),
				min_duration=duration, max_duration=duration)
			env = tools.wrappers.ConvertTo32Bit(env)
			envs.append(env)
		batch_env = tools.BatchEnv(envs, blocking=True)
		batch_env = tools.InGraphBatchEnv(batch_env)
		return batch_env
	
	def test_hoge(self):
		print(tf.random_uniform([5, 3]))


if __name__ == '__main__':
	tf.test.main()