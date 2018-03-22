"""Wrappers for OpenAI Gym environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import gym.spaces
import numpy as np
import tensorflow as tf

import sys
import multiprocessing
import traceback
import atexit


# import logging, coloredlogs
#
# logger = multiprocessing.log_to_stderr()
# logger.setLevel(logging.WARN)
#
# coloredlogs.install(level='WARN')
# coloredlogs.install(level='WARN', logger=logger)


class AutoReset(object):
	"""Automatically reset enviroment when the episode is done."""
	
	def __init__(self, env: gym.Env):
		self._env = env
		self._done = True
	
	def __getattr__(self, name):
		return getattr(self._env, name=name)
	
	def step(self, action):
		if self._done:
			observ, reward, done, info = self._env.reset(), 0., False, {}
		else:
			observ, reward, done, info = self._env.step(action)
		self._done = done
		return observ, reward, done, info
	
	def reset(self):
		self._done = False
		return self._env.reset()


class RangeNormalize(object):
	"""Normalize the specialized observation and action ranges to [-1, 1]."""
	
	def __init__(self, env, observ=None, action=None):
		self._env = env
		self._should_normalize_observ = (
				observ is not False and self._is_finite(self._env.observation_space))
		if observ is True and not self._should_normalize_observ:
			raise ValueError('Cannot normalize infinite observation range.')
		if observ is None and not self._should_normalize_observ:
			tf.logging.info('Not normalizing infinite observation range.')
		self._should_normalize_action = (
				action is not False and self._is_finite(self._env.action_space))
		if action is True and not self._should_normalize_action:
			raise ValueError('Cannot normalize infinite action range.')
		if action is None and not self._should_normalize_action:
			tf.logging.info('Not normalizing infinite action range.')
	
	def __getattr__(self, name):
		return getattr(self._env, name)
	
	@property
	def observation_space(self):
		space = self._env.observation_space
		if not self._should_normalize_observ:
			return space
		return gym.spaces.Box(-np.ones(space.shape), np.ones(space.shape))
	
	@property
	def action_space(self):
		space = self._env.action_space
		if not self._should_normalize_action:
			return space
		return gym.spaces.Box(-np.ones(space.shape), np.ones(space.shape))
	
	def step(self, action):
		if self._should_normalize_action:
			action = self._denormalize_action(action)
		observ, reward, done, info = self._env.step(action)
		if self._should_normalize_observ:
			observ = self._normalize_observ(observ)
		return observ, reward, done, info
	
	def reset(self):
		observ = self._env.reset()
		if self._should_normalize_observ:
			observ = self._normalize_observ(observ)
		return observ
	
	def _denormalize_action(self, action):
		min_ = self._env.action_space.low
		max_ = self._env.action_space.high
		action = (action + 1) / 2 * (max_ - min_) + min_
		return action
	
	def _normalize_observ(self, observ):
		min_ = self._env.observation_space.low
		max_ = self._env.observation_space.high
		observ = 2 * (observ - min_) / (max_ - min_) - 1
		return observ
	
	def _is_finite(self, space):
		return np.isfinite(space.low).all() and np.isfinite(space.high).all()


class ClipAction(object):
	"""Clip out of range actions to the action space of the environment."""
	
	def __init__(self, env):
		self._env = env
	
	def __getattr__(self, name):
		return getattr(self._env, name)
	
	@property
	def action_space(self):
		shape = self._env.action_space.shape
		return gym.spaces.Box(-np.inf * np.ones(shape), np.inf * np.ones(shape))
	
	def step(self, action):
		action_space = self._env.action_space
		action = np.clip(action, action_space.low, action.high)
		return self._env.step(action)


class LimitDuration(object):
	"""End episodes after specified number of steps.
	指定されたステップ数の後にエピソードを終了する。
	"""
	
	def __init__(self, env, duration):
		self._env = env
		self._duration = duration
		self._step = None
	
	def __getattr__(self, name):
		return getattr(self._env, name)
	
	def step(self, action):
		if self._step is None:
			raise RuntimeError('Must reset environment.')
		observ, reward, done, info = self._env.step(action)
		self._step += 1
		if self._step >= self._duration:
			done = True
			self._step = None
		return observ, reward, done, info
	
	def reset(self):
		self._step = 0
		return self._env.reset()


class ExternalProcess(object):
	"""Step environment in a separate process for lock free paralellism."""
	
	# Message type for communication via the pipe.
	_ACCESS = 1
	_CALL = 2
	_RESULT = 3
	_EXCEPTION = 4
	_CLOSE = 5
	
	def __init__(self, constructor):
		"""Step environment in a separate for lock free paralellism.

		The environment will be created in the external process by calling the
		specified callable. This can be an environment class, or a function
		creating the environment and potentially wrapping it. The returned
		environment should not access global variables.

		Args:
			constructor: Callable that creates and returns an OpenAI gym environment.


		Attributes:
			observation_space: The cached observation space of the environment.
			action_space: The cached action space of the environment.
		"""
		self._conn, conn = multiprocessing.Pipe()
		self._process = multiprocessing.Process(
			target=self._worker, args=(constructor, conn))
		atexit.register(self.close)
		self._process.start()
		self._observ_space = None
		self._action_space = None
	
	@property
	def observation_space(self):
		if not self._observ_space:
			self._observ_space = self.__getattr__('observation_space')
		return self._observ_space
	
	@property
	def action_space(self):
		if not self._action_space:
			self._action_space = self.__getattr__('action_space')
		return self._action_space
	
	def __getattr__(self, name):
		"""Request an attribute from the environment.

		Args:
			name: Attribute to access.

		Returns:
			Value of the attribute.
		"""
		self._conn.send((self._ACCESS, name))
		return self._receive()
	
	def call(self, name, *args, **kwargs):
		"""Asyncronously call a method of the external environment.

		Args:
			name: Name of the method to call.
			*args: Positional arguments to forward to the method.
			**kwargs: Keyword arguments to forward to the method.

		Returns:
			Promise object that blocks and provides the return value when called.
		"""
		payload = name, args, kwargs
		self._conn.send((self._CALL, payload))
		return self._receive
	
	def close(self):
		"""Send a close message to the external process and join it."""
		try:
			self._conn.send((self._CLOSE, None))
			self._conn.close()
		except IOError:
			# The connection was already closed.
			pass
		self._process.join()
	
	def step(self, action, blocking=True):
		"""Step the environment.

		Args:
			action: The action to apply to the environment.
			blocking: Whether to wait for the result.

		Returns:
			Transition tuple when blocking, otherwise callable that returns the
			transition tuple.
		"""
		promise = self.call('step', action)
		if blocking:
			return promise()
		else:
			return promise
	
	def reset(self, blocking=True):
		"""Reset the environment.

		Args:
			blocking: Whether to wait for the result.

		Returns:
			New observation when blocking, otherwise callable that returns the new
			observation.
		"""
		promise = self.call('reset')
		if blocking:
			return promise()
		else:
			return promise
	
	def _receive(self):
		"""Wait for a message from the worker process and return its payload.

		Raises:
			Exception: An exception was raised inside the worker process.
			KeyError: The reveived message is of an unknown type.
		Returns:
			Payload object of the message.
		"""
		message, payload = self._conn.recv()
		# Re-raise exceptions in the main process.
		if message == self._EXCEPTION:
			stacktrace = payload
			raise Exception(stacktrace)
		if message == self._RESULT:
			return payload
		raise KeyError('Received message of unexpected type {}'.format(message))
	
	def _worker(self, constructor, conn):
		"""The process waits for actions and sends back environment results.

		Args:
			constructor: Constructor for the OpenAI Gym environment.
			conn: Connection for communication tothe main process.

		Raises:
			KeyError: When receiving a message of unknown type.
		"""
		try:
			env = constructor()
			while True:
				try:
					# Only block for short times to have keyboard exception be raised.
					if not conn.poll(0.1):
						continue
					message, payload = conn.recv()
				except (EOFError, KeyboardInterrupt):
					break
				if message == self._ACCESS:
					name = payload
					result = getattr(env, name)
					conn.send((self._RESULT, result))
					continue
				if message == self._CALL:
					name, args, kwargs = payload
					result = getattr(env, name)(*args, **kwargs)
					conn.send((self._RESULT, result))
					continue
				if message == self._CLOSE:
					assert payload is None
					break
				raise KeyError('Received message of unknown type {}.'.format(message))
		except Exception:  # pylint: disable=broad-except
			stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
			tf.logging.error('Error in environment process: {}'.format(stacktrace))
			conn.send((self._EXCEPTION, stacktrace))
		conn.close()


class ConvertTo32Bit(object):
	"""Convert data types of an OpenAI Gym environment to 32 bit."""
	
	def __init__(self, env):
		"""Convert data types of an OpenAI Gym environment to 32 bit.

		Args:
			env: OpenAI Gym environment.
		"""
		self._env = env
	
	def __getattr__(self, name):
		"""Forward unimplemented attributes to the original environment.

		Args:
			name: Attribute that was accessed.

		Returns:
			Value behind the attribute name in the wrapped environment.
		"""
		return getattr(self._env, name)
	
	def step(self, action):
		"""Forward action to the wrapped environment.

		Args:
			action: Action to apply to the environment.

		Raises:
			ValueError: Invalid action.

		Returns:
			Converted observation, converted reward, done flag, and info object.
		"""
		observ, reward, done, info = self._env.step(action)
		observ = self._convert_observ(observ)
		reward = self._convert_reward(reward)
		return observ, reward, done, info
	
	def reset(self):
		"""Reset the environment and convert the resulting observation.

		Returns:
			Converted observation.
		"""
		observ = self._env.reset()
		observ = self._convert_observ(observ)
		return observ
	
	def _convert_observ(self, observ):
		"""Convert the observation to 32 bits.

		Args:
			observ: Numpy observation.

		Raises:
			ValueError: Observation contains infinite values.

		Returns:
			Numpy observation with 32-bit data type.
		"""
		if np.isinf(observ).any():
			raise ValueError('Infinite observation encountered.')
		if observ.dtype == np.float64:
			return observ.astype(np.float32)
		if observ.dtype == np.int64:
			return observ.astype(np.int32)
		return observ
	
	def _convert_reward(self, reward):
		"""Convert the reward to 32 bits.

		Args:
			reward: Numpy reward.

		Raises:
			ValueError: Reward contains infinite values.

		Returns:
			Numpy reward with 32-bit data type.
		"""
		if not np.isfinite(reward).all():
			raise ValueError('Infinite reward encountered.')
		return np.array(reward, dtype=np.float32)