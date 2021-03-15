import numpy as np
import matplotlib.pyplot as plt
from queue import Queue

np.random.seed(11)
c = np.random.randn(1)
print(f'constant: {c}')

rho = np.random.rand(3)
print(f'rho: {rho}')
rho1, rho2 = rho[0], rho[-1]

def ar_elem(constant, rho1, rho2, Nth):
    que = Queue()

    x1 = rho1 * 1 + 0.1 * np.random.rand(1)  # constant + rho*1 + bias(0,1)
    que.put(x1)

    x2 = rho1 * x1 + rho2 * 1 + 0.1 * np.random.rand(1)
    que.put(x2)


    if Nth == 1:
        return x1
    elif Nth == 2:
        return x2
    else:
        while Nth:
        # print(f'Nth trial: {Nth}')
        ar_obj = constant + rho1 * que.get() + rho2 * que.get() + 0.1 * np.random.rand(1)
        que.put(ar_obj)

    # print(ar_obj, rho, np.random.rand(1))
        Nth -= 1
        return ar_obj


t = ar_elem(1, 0.5, 0.2, 2)
print(t)

for i in range(3):
    print(ar_elem(1, 0.5, 0.2, i))
