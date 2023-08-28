import numpy as np

a = np.zeros((5,8))

a = np.vstack([a, np.ones(8)])
a = np.delete(a, 0, 0)
print(a)
