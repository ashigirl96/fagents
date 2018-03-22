r"""Script to train a batch reinforcement learning algorithm.

Command line:

	python -m agents.scripts.train --logdir=/path/to/logdir --config=pendulum
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from fagents import tools


def _create_environment(config):
	"""Constructor for an instance of the environment.
	
	Args:
		config: Object providing configurations via attributes.

	Returns:
		Wrapped OpenAI Gym environment.
	"""
	if isinstance(config.env, str):
		env = gym.make(config.env)
	else:
		env = config.env()
	if config.max_length:
		env = tools.wrappers.LimitDuration(env, config.max_length)
	env = tools.wrappers.RangeNormalize(env)
	env = tools.wrappers.ClipAction(env)
	env = tools.wrappers.ConvertTo32Bit(env)
	return env