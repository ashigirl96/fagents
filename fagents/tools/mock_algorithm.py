"""Mock algorithm for testing reinforcement learning code."""

import tensorflow as tf


class MockAlgorithm(object):
  """Produce random actions and empty summaries."""

  def __init__(self, envs):
    """Produce random actions and empty summaries.

    Args:
      envs: List of in-graph environments.
    """
    self._envs = envs

  def begin_episode(self, unused_agent_indices):
    return tf.constant("")

  def perform(self, agent_indices, unused_observ):
    shape = (tf.shape(agent_indices)[0],) + self._envs[0].action_space.shape
    low = self._envs[0].action_space.low
    high = self._envs[0].action_space.high
    action = tf.random_uniform(shape) * (high - low) + low
    return action, tf.constant("")

  def experience(self, unused_agent_indices, *unused_transition):
    return tf.constant("")

  def end_episode(self, unused_agent_indices):
    return tf.constant("")