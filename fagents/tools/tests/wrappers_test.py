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
    env.close()

  def test_close_no_hang_after_step(self):
    constructor = functools.partial(
      tools.MockEnvironment,
      observ_shape=(2, 3), action_shape=(2,),
      min_duration=5, max_duration=5)
    env = tools.wrappers.ExternalProcess(constructor)
    env.reset()
    env.step(env.action_space.sample())
    env.step(env.action_space.sample())
    env.close()

  def test_reraise_exception_in_init(self):
    constructor = MockEnvironmentCrashInInit
    env = tools.wrappers.ExternalProcess(constructor)
    with self.assertRaises(Exception):
      env.reset()

  def test_reraise_exception_in_step(self):
    constructor = functools.partial(MockEnvironmentCrashInStep, crash_at_step=3)
    env = tools.wrappers.ExternalProcess(constructor)
    env.reset()
    env.step(env.action_space.sample())
    env.step(env.action_space.sample())
    with self.assertRaises(Exception):
      env.step(env.action_space.sample())


class MockEnvironmentCrashInInit(object):
  """Raise an error when instantiated."""

  def __init__(self, *unused_args, **unused_kwargs):
    raise RuntimeError()


class MockEnvironmentCrashInStep(tools.MockEnvironment):
  """Raise an error when instantiated."""

  def __init__(self, crash_at_step):
    super(MockEnvironmentCrashInStep, self).__init__(
      observ_shape=(2, 3), action_shape=(2,),
      min_duration=crash_at_step + 1, max_duration=crash_at_step + 1)
    self._crash_at_step = crash_at_step

  def step(self, *args, **kwargs):
    transition = super(MockEnvironmentCrashInStep, self).step(*args, **kwargs)
    if self.steps[-1] == self._crash_at_step:
      raise RuntimeError()
    return transition


if __name__ == '__main__':
  tf.app.run()