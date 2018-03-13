"""Test an OpenAI Gym batch environment into the TensorFlow Graph."""

from fagents import tools
import functools

import tensorflow as tf
import gym


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

  def test_batch_getattr(self):
    constructor = functools.partial(
      tools.MockEnvironment,
      observ_shape=(2, 3), action_shape=(4,),
      min_duration=5, max_duration=5)
    batch_env = _define_batch_env(constructor, 5, True, False)
    self.assertEqual(batch_env.observ.shape, (5, 2, 3), msg="observ.shape")
    self.assertEqual(batch_env.action.shape, (5, 4), msg="action.shape")

  def test_batch_reset(self):
    constructor = functools.partial(
      tools.MockEnvironment,
      observ_shape=(2, 3), action_shape=(4,),
      min_duration=5, max_duration=5)
    batch_env = _define_batch_env(constructor, 5, True, False)
    observ = batch_env.reset()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      init.run()
      sess.run(observ)


if __name__ == '__main__':
  tf.app.run()