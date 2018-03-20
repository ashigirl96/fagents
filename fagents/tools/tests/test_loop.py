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
	
	def test_phases_feed(self):
		score = tf.placeholder(tf.float32, [])
		loop = tools.Loop(None)
		loop.add_phase(
			'phase_1', done=True, score=score, summary='', steps=1, report_every=1,
			log_every=None, checkpoint_every=None, feed={score: 1})
		loop.add_phase(
			'phase_2', done=True, score=score, summary='', steps=3, report_every=1,
			log_every=None, checkpoint_every=None, feed={score: 2})
		loop.add_phase(
			'phase_3', done=True, score=score, summary='', steps=2, report_every=1,
			log_every=None, checkpoint_every=None, feed={score: 3})
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			scores = list(loop.run(sess, saver=None, max_step=15))
		self.assertAllEqual([1, 2, 2, 2, 3, 3, 1, 2, 2, 2, 3, 3, 1, 2, 2], scores)
	
	def test_phases_feed2(self):
		import numpy as np
		score = tf.placeholder(tf.float32, [])
		loop = tools.Loop(None)
		loop.add_phase(
			'phase_1', done=True, score=score, summary='', steps=1, report_every=1,
			log_every=None, checkpoint_every=None, feed={score: 1})
		loop.add_phase(
			'phase_2', done=True, score=score, summary='', steps=3, report_every=1,
			log_every=None, checkpoint_every=None, feed={
				score: np.random.randn(1)[0]
			})
		loop.add_phase(
			'phase_3', done=True, score=score, summary='', steps=2, report_every=1,
			log_every=None, checkpoint_every=None, feed={score: 3})
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			scores = list(loop.run(sess, saver=None, max_step=15))
		self.assertAllEqual([1, 2, 2, 2, 3, 3, 1, 2, 2, 2, 3, 3, 1, 2, 2], scores)
	
	def test_average_score_over_phases(self):
		loop = tools.Loop(None)
		loop.add_phase(
			'phase_1', done=True, score=1, summary='', steps=1, report_every=2)
		loop.add_phase(
			'phase_2', done=True, score=2, summary='', steps=2, report_every=5)
		# Score:    1 2 2 1 2 2 1 2 2 1 2 2 1 2 2 1 2
		# Report 1:       x           x           x
		# Report 2:               x             x
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			scores = list(loop.run(sess, saver=None, max_step=17))
		
		self.assertAllEqual([1, 2, 1, 2, 1], scores)
	
	def test_average_score_over_phases2(self):
		loop = tools.Loop(None)
		loop.add_phase(
			'phase_1', done=True, score=1, summary='', steps=1, report_every=2)
		# loop.add_phase(
		# 	'phase_2', done=True, score=2, summary='', steps=2, report_every=5)
		# Score:    1 2 2 1 2 2 1 2 2 1 2 2 1 2 2 1 2
		# Report 1:       x           x           x
		# Report 2:               x             x
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			scores = list(loop.run(sess, saver=None, max_step=17))
		
		self.assertAllEqual([1, 1, 1], scores)
	
	def test_average_score_over_phases3(self):
		loop = tools.Loop(None)
		loop.add_phase(
			'phase_1', done=True, score=1, summary='', steps=1, report_every=2)
		loop.add_phase(
			'phase_2', done=True, score=2, summary='', steps=2, report_every=5)
		loop.add_phase(
			'phase_2', done=True, score=3, summary='', steps=1, report_every=5)
		# Score:    1 2 2 3 1 2 2 3 1 2 2 3 1 2 2 1 2 2 1 2
		# Report 1:         x               x           x
		# Report 2:                   x               x
		# Report 3:
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			scores = list(loop.run(sess, saver=None, max_step=17))
		
		self.assertAllEqual([1, 2, 1], scores)
	
	def test_not_done(self):
		step = tf.Variable(0, False, dtype=tf.int32, name='step')
		done = tf.equal((step + 1) % 2, 0)
		score = tf.cast(step, tf.float32)
		loop = tools.Loop(None, step)
		loop.add_phase(
			'phase_1', done, score, summary='', steps=1, report_every=3)
		# Step:   0 1 2 3 4 5 6 7 8 9
		# Score:  0 1 2 3 4 5 6 7 8 9 10 11
		# Done:     x   x   x   x   x     x
		# Report:     x     x     x       x
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			scores = list(loop.run(sess, saver=None, max_step=12))
		self.assertAllEqual([1, 4, 7, 10], scores)
	
	def test_not_done_batch(self):
		step = tf.Variable(0, False, dtype=tf.int32, name='step')
		done = tf.equal([step % 3, step % 4], 0)
		score = tf.cast([step, step ** 2], tf.float32)
		loop = tools.Loop(None, step)
		loop.add_phase(
			'phase_1', done, score, summary='', steps=1, report_every=8)
		# Step:    0  2  4  6  8
		# Score 1: 0  2  4  6
		# Done 1:  x        x
		# Score 2: 0  4 16 32
		# Done 2:  x     x
		# Report :             x
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			scores = list(loop.run(sess, saver=None, max_step=8))
			self.assertEqual(8, sess.run(step))
		self.assertAllEqual([(0 + 0 + 16 + 6) / 4], scores)


if __name__ == '__main__':
	tf.test.main()