import functools

def f(foo=None):
  print(foo)


def main():
  bar = functools.partial(f, "bar")
  bar()
  hoge = functools.partial(f, foo="hoge")
  hoge()


if __name__ == '__main__':
  main()