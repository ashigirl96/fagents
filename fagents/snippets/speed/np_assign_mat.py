from numba import jit
import numpy as np
import time


@jit
def func():
  foo = np.random.uniform(0, 1, size=(1000, 3000))
  start_time = time.time()
  for i in range(1000):
    bar = np.random.uniform(0, 1, size=(3000, 3000))
    foo = foo @ bar
  process_time = time.time() - start_time
  return process_time


if __name__ == '__main__':
  print(func())