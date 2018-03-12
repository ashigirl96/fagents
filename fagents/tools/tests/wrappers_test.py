"""Tests for environment wrappers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf

from fagents import tools


class ExternalProcessTest(tf.test.TestCase):

  def test_close_no_hang_after_init(self):
    constructor = functools.partial(
      tools.MockEnvironment,
      observ_shape=(2, 3), action_shape=(2,),
      min_duration=5, max_duration=5)
    env = tools.wrappers.ExternalProcess(constructor)
    env.reset()
    env.step(env.action_space.sample())
    env.step(env.action_space.sample())
    env.close()


if __name__ == '__main__':
  tf.app.run()