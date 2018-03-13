"""Test an OpenAI Gym batch environment into the TensorFlow Graph."""

from agents import tools
from fagents.tools import MockEnvironment
import functools

import tensorflow as tf


def _define_batch_env(constructor, num_agents, env_processes, blocking):
  if env_processes:
    envs = [
      tools.wrappers.ExternalProcess(constructor)
      for _ in range(num_agents)]
  else:
    envs = [constructor() for _ in range(num_agents)]
  batch_env = tools.BatchEnv(envs, blocking=blocking)
  batch_env = tools.InGraphBatchEnv(batch_env)
  return batch_env


class TestInGraphBatchEnv(tf.test.TestCase):

  def setUp(self):
    constructor = functools.partial(
      MockEnvironment,
      observ_shape=(2, 3), action_shape=(4,),
      min_duration=10, max_duration=10)
    self.batch_env = _define_batch_env(constructor, 5, True, False)

  def test_batch_getattr(self):
    self.assertEqual(self.batch_env.observ.shape, (5, 2, 3), msg="observ.shape")
    self.assertEqual(self.batch_env.action.shape, (5, 4), msg="action.shape")

  def test_batch_reset(self):
    observ = self.batch_env.reset()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      init.run()
      self.assertEqual(sess.run(observ).shape, (5, 2, 3))

  def test_batch_step(self):
    """For test different observ of first step and second step."""
    observ = self.batch_env.reset()
    step = self.batch_env.simulate(self.batch_env.action)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
      init.run()
      sess.run(observ)
      sess.run(step)
      first_step = sess.run(self.batch_env.observ)
      sess.run(step)
      second_step = sess.run(self.batch_env.observ)
      self.assertFalse((first_step == second_step).all())

  def test_batch_close(self):
    step = self.batch_env.simulate(self.batch_env.action)
    init = tf.global_variables_initializer()
    with self.assertRaises(Exception):
      with tf.Session() as sess:
        init.run()

        sess.run(self.batch_env.reset())
        self.batch_env.close()
        sess.run(step)


if __name__ == '__main__':
  tf.app.run()