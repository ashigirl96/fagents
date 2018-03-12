"""Snippets for how to code multi process"""

import multiprocessing
import time

def _worker(i):

  print("I'm {0}'th worker".format(i))
  time.sleep(1)
  return


def f(conn):
  conn.send([42, None, 'hello'])
  conn.close()

def main1():
  parent_conn, child_conn = multiprocessing.Pipe()
  p = multiprocessing.Process(target=f, args=(child_conn,))
  p.start()
  print(parent_conn.recv())   # prints "[42, None, 'hello']"


def main2():
  parent_conn, child_conn = multiprocessing.Pipe()
  p = multiprocessing.Process(target=f, args=(child_conn,))
  p.start()
  print(parent_conn.recv())   # prints "[42, None, 'hello']"


if __name__ == '__main__':
  main2()

# def main():
#   print("Start...")
#   for i in range(10):
#     process = multiprocessing.Process(target=_worker, args=(i,))
#     process.start()
#
#
#
# if __name__ == '__main__':
#   main()