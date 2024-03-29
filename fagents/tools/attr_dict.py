"""Wrap a dictionary to access keys as attributes."""

import contextlib


class AttrDict(dict):

  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    super(AttrDict, self).__setattr__('__mutable', False)

  def __getattr__(self, key):
    # Do not provide None for unimplemented magic attributes.
    if key.startswith('__'):
      raise AttributeError
    return self.get(key, None)

  def __setattr__(self, key, value):
    if not self._mutable:
      message = "Cannot set attribute '{}'.".format(key)
      message += "Use obj.unlocked: scope to set attributes."
      raise RuntimeError(message)
    if key.startswith('__'):
      raise AttributeError("Cannot set magic attribute '{}'.".format(key))
    self[key] = value

  @property
  @contextlib.contextmanager
  def unlocked(self):
    super(AttrDict, self).__setattr__('_mutable', True)
    yield
    super(AttrDict, self).__setattr__('_mutable', False)

  def copy(self):
    return type(self)(super(AttrDict, self).copy())
