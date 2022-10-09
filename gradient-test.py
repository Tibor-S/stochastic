import numpy as np


g = np.gradient([1, 4, 1],)
h = np.gradient([[4, 4, 4, 4, 4],
                 [4, 1, 1, 1, 4],
                 [4, 1, 0, 1, 4],
                 [4, 1, 1, 1, 4],
                 [4, 4, 4, 4, 4]])
for h1 in h:
    print(h1)
# print(np.matmul(h[0], h[1]))
vector = np.array([1, 2, 3])
print(np.divide(np.dot(vector, vector), 2))
