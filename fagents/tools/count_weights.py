"""Count learnable parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import numpy as np
import tensorflow as tf


def count_weights(scope=None, exclude=None, graph=None):
  """Count learnable parameters.

  Args:
    scope: Restrict the count to a variable scope.
    exclude: Regex to match variable names to exclude.
    graph: Operate on graph other than the current default graph.

  Returns:
    Number of learnable parameters as integer.
  """
  if scope:
    scope = scope if scope.endswith('/') else scope + '/'
  graph = graph or tf.get_default_graph()
  vars_ = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  if scope:
    vars_ = [var for var in vars_ if var.name.startswith(scope)]
  if exclude:
    exclude = re.compile(exclude)
    vars_ = [var for var in vars_ if not exclude.match(var.name)]
  shapes = [var.get_shape().as_list() for var in vars_]
  return int(sum(np.prod(shape) for shape in shapes))
