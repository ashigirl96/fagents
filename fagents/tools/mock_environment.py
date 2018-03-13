"""Mock environment for testing reinforcement learning code."""

import gym
import gym.spaces
import numpy as np


class MockEnvironment(object):
  """Generate random agent input and keep track of statistics."""

  def __init__(self, observ_shape, action_shape, min_duration, max_duration):
    """Generate random agent input and keep track of statistics.

    Args:
      observ_shape: Shape for the random observations.
      action_shape: Shape for the action space.
      min_duration: Minimum number of steps per episode.
      max_duration: Maximum number of steps per episode..

    Attributes:
      steps: List of actual simulated lengths for all episodes.
      durations: List of decided lengths for all episodes.
    """
    self._observ_shape = observ_shape
    self._action_shape = action_shape
    self._min_duration = min_duration
    self._max_duration = max_duration
    self._random = np.random.RandomState(0)
    self.steps = []
    self.durations = []

  @property
  def observation_space(self):
    low = np.zeros(self._observ_shape)
    high = np.ones(self._observ_shape)
    return gym.spaces.Box(low, high, dtype=np.float32)

  @property
  def action_space(self):
    low = np.zeros(self._action_shape)
    high = np.ones(self._action_shape)
    return gym.spaces.Box(low, high, dtype=np.float32)

  @property
  def unwrapped(self):
    return self

  def step(self, action):
    assert self.action_space.contains(action)
    assert self.steps[-1] < self.durations[-1]
    self.steps[-1] += 1
    observ = self._current_observation()
    reward = self._current_reward()
    done = self.steps[-1] >= self.durations[-1]
    info = dict()
    return observ, reward, done, info

  def reset(self):
    duration = self._random.randint(self._min_duration, self._max_duration + 1)
    self.steps.append(0)
    self.durations.append(duration)
    return self._current_observation()

  def _current_observation(self):
    return self._random.uniform(0, 1, self._observ_shape).astype(np.float32)

  def _current_reward(self):
    return self._random.uniform(-1, 1).astype(np.float32)
