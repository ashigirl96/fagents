"""tests for script utility"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from fagents.scripts.utility import define_batch_env
import tensorflow as tf
from fagents.tools.mock_environment import MockEnvironment
import functools


class DefineBatchEnvTest(tf.test.TestCase):
  def test_define_batch_env(self):
    constructor = functools.partial(
      MockEnvironment,
      observ_shape=(2, 3), action_shape=(2,),
      min_duration=5, max_duration=5)
    envs = define_batch_env(constructor, 3, True)
    print(envs[0]._conn)  # <multiprocessing.connection.Connection object at 0x7fa813c35860>
    print(envs[1]._conn)  # <multiprocessing.connection.Connection object at 0x7fa813bc92e8>


if __name__ == '__main__':
  tf.test.main()