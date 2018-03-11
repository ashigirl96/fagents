import tensorflow as tf

from fagents.tools import count_weights


class CountWeightsTest(tf.test.TestCase):

  def test_trainable_vars(self):
    foo = tf.Variable(13)
    bar = tf.Variable(tf.random_uniform((2, 3)))
    global_steps = tf.Variable(0, trainable=False)
    self.assertEqual(count_weights(), 7)

  def test_trainable_vars_in_scope(self):
    with tf.name_scope('models'):
      foo = tf.Variable([13, 10, 20], name='foo')
      bar = tf.Variable([42, 20])
    dog = tf.Variable([57, 30])

    self.assertTrue("models/foo" in foo.name)
    self.assertEqual(count_weights(scope="models"), 5)
    self.assertEqual(count_weights(), 7)

  def test_exclude(self):
    with tf.variable_scope('models'):
      inputs = tf.placeholder(tf.float32, [None, 24, 24, 3])
      x = tf.layers.conv2d(inputs, 3, 3)
      x = tf.layers.flatten(x)
      x = tf.layers.dense(inputs, 10)
    self.assertNotEqual(count_weights(scope='models', exclude=r'.*/conv2d/.*'),
                        count_weights(scope='models'))

  def test_non_default_graph(self):
    graph = tf.Graph()
    with graph.as_default():
      tf.Variable(tf.zeros((2, 3)), trainable=True)
      tf.Variable(tf.zeros((8, 2)), trainable=False)
    self.assertNotEqual(graph, tf.get_default_graph)
    self.assertEqual(count_weights(graph=graph), 6)


if __name__ == '__main__':
  tf.app.run()
