import numpy as np
import matplotlib.pyplot as plt

def myAR2(seed, N):
    """
    hw 1) rho1 rho2 relation specify
    hw 2) decoder LSTM (how it works?)
    
    :param seed:
    :param N:
    :return:
    """
    np.random.seed(seed)

    rho = np.random.uniform(-1, 1, (1)).round(3)
    rho2 = np.random.uniform(-1, 1, (1)).round(3)

    stk = 10 * np.array(np.random.randn(1).round(3))
    stk = np.append(stk, stk[-1]*rho+0.1*np.random.randn(1).round(3))
    print(stk)   # 2 components
    while N >0:
        yesterday = stk[-2].round(3)
        today = stk[-1].round(3)
        tomorrow = rho*yesterday + rho2*today + 0.1*np.random.randn(1).round(3)
        new_stk = np.append(stk, tomorrow)
        stk = new_stk
        N-=1

    plt.plot(range(len(stk)), stk)
    plt.show()
    return stk

print(myAR2(13, N=10))  # len 12

