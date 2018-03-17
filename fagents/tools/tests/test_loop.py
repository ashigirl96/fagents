"""Tests for the training loop."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from fagents import tools


class LoopTest(tf.test.TestCase):
  def test_report_every_step(self):
    step = tf.Variable(0, trainable=False, dtype=tf.int32, name='step')
    loop = tools.Loop(None, step)
    loop.add_phase(
      'phase_1', done=True, score=0, summary='', steps=1, report_every=3)
    # Step:  0 1 2 3 4 5 6 7 8
    # Report:    x     x     x
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      scores = loop.run(sess, saver=None, max_step=9)
      next(scores)
      self.assertEqual(3, sess.run(step))
      next(scores)
      self.assertEqual(6, sess.run(step))
      next(scores)
      self.assertEqual(9, sess.run(step))


if __name__ == '__main__':
  tf.test.main()