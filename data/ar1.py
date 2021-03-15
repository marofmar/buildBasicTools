import numpy as np
import matplotlib.pyplot as plt

np.random.seed(11)
c = np.random.randn(1)  # random constant from a normal distribution
print(f'constant: {c}')

rho = np.random.rand(3)  # random number from  a uniform distribution (positive as the range is from zero to one)
print(f'rho: {rho}')





def ar_elem(constant, rho, Nth):
    ar_obj = constant + rho * 1 + 0.1 * np.random.rand(1)  # constant + rho*1 + bias(0,1)
    # print(ar_obj)
    while Nth > 0:
        # print(f'Nth trial: {Nth}')
        ar_obj = ar_obj * rho + 0.1 * np.random.rand(1)
        # print(ar_obj, rho, np.random.rand(1))
        Nth -= 1
    ar_elem = ar_obj
    return float(ar_elem)


def show_ar(N):
    tmp = []
    sum = 0
    for i in range(N):
        tmp.append(ar_elem(c, rho[0], i + 5))  # using the random Constant, and random RHO
        sum += 1
    print(sum)

    print(tmp)

    plt.plot(range(N), tmp)
    plt.show()


