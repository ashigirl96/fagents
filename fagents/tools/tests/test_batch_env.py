"""Test combine multiple environments to step them in batch."""

import tensorflow as tf
import functools

from fagents import tools
import time


def _define_batch_env(constructor, num_agents, env_processes, blocking):
  if env_processes:
    envs = [
      tools.wrappers.ExternalProcess(constructor)
      for _ in range(num_agents)]
  else:
    envs = [constructor() for _ in range(num_agents)]
  batch_env = tools.BatchEnv(envs, blocking=blocking)
  return batch_env


def timer(func):
  start_time = time.time()
  func()
  return time.time() - start_time


class BatchEnvTest(tf.test.TestCase):
  def test_batch_blocking(self):
    constructor = functools.partial(
      MockEnvironmentHeavyProcess,
      process_time=1.5,
      observ_shape=(2, 3), action_shape=(2,),
      min_duration=5, max_duration=5)
    envs = _define_batch_env(constructor, 4, True, True)
    t = timer(envs.reset)
    self.assertAllClose(t, 1.5 * 4, rtol=0.1)

  def test_batch_unblocking(self):
    constructor = functools.partial(
      MockEnvironmentHeavyProcess,
      process_time=1.5,
      observ_shape=(2, 3), action_shape=(2,),
      min_duration=5, max_duration=5)
    envs = _define_batch_env(constructor, 4, True, False)
    t = timer(envs.reset)
    self.assertAllClose(t, 1.5, rtol=0.1)


class MockEnvironmentHeavyProcess(tools.MockEnvironment):

  def __init__(self, process_time, *args, **kwargs):
    self.process_time = process_time
    super(MockEnvironmentHeavyProcess, self).__init__(*args, **kwargs)

  def reset(self):
    observ = super(MockEnvironmentHeavyProcess, self).reset()
    time.sleep(self.process_time)
    return observ

  def step(self, action):
    observ = super(MockEnvironmentHeavyProcess, self).step(action=action)
    time.sleep(self.process_time)
    return observ


if __name__ == '__main__':
  tf.test.main()