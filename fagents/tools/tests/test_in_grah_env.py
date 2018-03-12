"""Test an OpenAI Gym environment into the TensorFlow Graph."""

from fagents.tools import InGraphEnv
from agents.tools import InGraphEnv as I
I.step
# from agents.tools import InGraphEnv

import tensorflow as tf
import gym


class TestInGraphEnv(tf.test.TestCase):

  def test_getattr(self):
    original_env = gym.make('Pendulum-v0')
    env = InGraphEnv(original_env)
    self.assertEqual(original_env.spec, env.spec)

  def test_reset(self):
    print()
    original_env = gym.make('Pendulum-v0')
    env = InGraphEnv(original_env)
    observ = env.reset()
    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      _observ = sess.run(observ)

  def test_simulate(self):
    original_env = gym.make('Pendulum-v0')
    env = InGraphEnv(original_env)
    step = env.simulate(action=env.action)
    print(step)


if __name__ == '__main__':
  tf.app.run()