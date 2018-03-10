import tensorflow as tf

from fagents.tools import attr_dict


class AttrDictTest(tf.test.TestCase):

  def test_construct_from_dict(self):
    initial = dict(foo=13, bar=42)
    obj = attr_dict.AttrDict(initial)
    self.assertEqual(obj.foo, 13)
    self.assertEqual(obj.bar, 42)

  def test_immutable_modify(self):
    obj = attr_dict.AttrDict(foo=13)
    with self.assertRaises(RuntimeError):
      obj.foo = 42

  def test_immutable_unlocked(self):
    obj = attr_dict.AttrDict()
    with obj.unlocked:
      obj.foo = 42
    self.assertEqual(obj.foo, 42)


if __name__ == '__main__':
  tf.app.run()
