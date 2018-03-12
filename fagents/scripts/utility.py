"""Utilitiy for using reinforcement learning algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import re

import ruamel.yaml as yaml
import tensorflow as tf

from fagents import tools


def define_batch_env(constructor, num_agents, env_processes):
  """Create environments and apply all desired w

  Args:
    constructor: Constructor of an OpenAI gym environment.
    num_agents: Number of environments to combine in the batch.
    env_processes: Whether to step environment in external processes.

  Returns:
    In-graph environments object.
  """
  with tf.variable_scope('environments'):
    if env_processes:
      envs = [
        tools.wrappers.ExternalProcess(constructor)
        for _ in range(num_agents)]
    else:
      envs = [constructor() for _ in range(num_agents)]
    batch_env = tools.BatchEnv(envs, blocking=not env_processes)
    # batch_env = tools.InGraphBatchEnv(batch_env)
  return batch_env