"""Compute a streaming estimation of the mean of submitted tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class StreamingMean(object):
  """Compute a streaming estimation of the mean of submitted tensors."""

  def __init__(self, shape, dtype):
    """Specify the shape and dtype of the mean to be estimated.

    Note that a float mean to zero submitted elements is NaN, while computing
    the integer mean of zero elements raises a division by zero error.

    Args:
      shape(tuple): Shape of the mean to compute.
      dtype(tf.Dtype): Data type of the mean to compute.

    """
    self._dtype = dtype
    self._sum = tf.Variable(lambda: tf.zeros(shape, dtype), False)
    self._count = tf.Variable(lambda: 0, trainable=False)

  @property
  def value(self):
    """The current value of the mean."""
    return  self._sum / tf.cast(self._count, self._dtype)

  @property
  def count(self):
    """The number of submitted samples."""
    return self._count

  def submit(self, value: tf.Tensor):
    """Submit a single or batch tensor to refine the streaming mean."""
    if value.shape.ndims == self._sum.shape.ndims:
      value = value[tf.newaxis, ...]
    return tf.group(
      self._sum.assign_add(tf.reduce_sum(value, 0)),
      self._count.assign_add(tf.shape(value)[0]))

  def clear(self):
    """Return the mean estimate and reset the streaming statistics."""
    value = self._sum / tf.cast(self._count, self._dtype)
    with tf.control_dependencies([value]):
      reset_value = self._sum.assign(tf.zeros_like(self._sum))
      reset_count = self._count.assign(0)
    with tf.control_dependencies([reset_value, reset_count]):
      return tf.identity(value)
