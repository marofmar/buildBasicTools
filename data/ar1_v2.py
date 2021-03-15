import numpy as np
import matplotlib.pyplot as plt

def myAR1(seed, N):
    np.random.seed(seed)
    stk = 10 * np.array(np.random.randn(1).round(3))
    rho = np.random.uniform(-1, 1, (1)).round(3)
    while N >0:
        now = stk[-1].round(3)
        next = rho * now + 0.1*np.random.randn(1).round(3)  # add bias
        new_stk = np.append(stk, next)
        stk = new_stk
        N-=1

    plt.plot(range(len(stk)), stk)
    plt.show()
    return stk

print(myAR1(13, N =10))  # len 11



# stk = 10*np.array(np.random.randn(1).round(3), dtype= float)
# print(f"STACK: {stk}")  # starting point
#
# rho = np.random.uniform(-1, 1, (1)).round(3)
# print(f"RHO: {rho}")
#
# bias = 0.1 * np.random.randn(1).round(3)
# print(f"BIAS: {bias}")
#
# n = 10
# while n>0:
#     Xn = stk[-1].round(3)
#     print(f"Xn: {Xn}")
#     Xn_1 = (Xn * rho + bias).round(3)
#     print(f"{Xn} x {rho} + {bias} = {Xn_1}")
#     stk = np.append(stk, Xn_1)
#     n-=1
#
# print(stk)
#
# plt.plot(range(11), stk)
# plt.show()