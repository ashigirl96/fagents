"""Test for testing Mock environment."""

from fagents import tools
import tensorflow as tf
import functools
import numpy as np


class MockEnvTest(tf.test.TestCase):
  def test_action_dtype(self):
    constructor = functools.partial(
      tools.MockEnvironment,
      observ_shape=(2, 3), action_shape=(4,),
      min_duration=5, max_duration=5)
    env = constructor()
    env.reset()
    dtype = env.action_space.dtype
    self.assertEqual(dtype, np.float32)


if __name__ == '__main__':
  tf.test.main()