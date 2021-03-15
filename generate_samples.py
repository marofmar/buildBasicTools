import numpy as np
import pandas as pd

from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima.model import ARIMA

import matplotlib.pyplot as plt

def generate_ar2_ma2(p, q, n):
    """
    :param p: AR(p) p in the form of [a, b]
    :param q: MA(q) q in the form of [c, d]
    :param n: number of samples
    :return: series of numbers in numpy array type
    """
    arparams = np.array([p])
    maparams = np.array([q])

    arparams = np.r_[1, -arparams]
    maparams = np.r_[1, -maparams]

    samples = arma_generate_sample(arparams, maparams, n)  # generated sample

    print(samples)
    #print(type(ar_sample)) # numpy array

    #ar_mod = ARIMA(ar_sample, order = (1, 0, 0))  # model
    #print(ar_mod)


    #ar_res = ar_mod.fit()
    #print(ar_res.summary())

    return samples


def generate_ar1_ma1(p, q, n):
    """
    :param p: AR(p)
    :param q: MA(q)
    :param n: number of samples
    :return: series of numbers in numpy array type
    """
    arparams = np.array([p])
    maparams = np.array([q])

    arparams = np.r_[1, -arparams]
    maparams = np.r_[1, -maparams]

    samples = arma_generate_sample(arparams, maparams, n)  # generated sample

    print(samples)
    #print(type(ar_sample)) # numpy array

    #ar_mod = ARIMA(ar_sample, order = (1, 0, 0))  # model
    #print(ar_mod)


    #ar_res = ar_mod.fit()
    #print(ar_res.summary())

    return samples

generate_ar1_ma1(0.5, 0, 10)
generate_ar1_ma1(0, 0.5, 10)

def generate_ar1(p, n):
    arparams = np.array([p])
    maparams = np.array([0.0])

    arparams = np.r_[1, -arparams]
    maparams = np.r_[1, maparams]

    ar_sample = arma_generate_sample(arparams, maparams, n)  # generated sample

    print(ar_sample)
    print(type(ar_sample)) # numpy array

    ar_mod = ARIMA(ar_sample, order = (1, 0, 0))  # model
    #print(ar_mod)


    #ar_res = ar_mod.fit()
    #print(ar_res.summary())

    return ar_sample

def generate_ma1(p, n):
    arparams = np.array([0.0])
    maparams = np.array([p])

    arparams = np.r_[1, -arparams]
    maparams = np.r_[1, -maparams]

    ma_sample = arma_generate_sample(arparams, maparams, n)  # generated sample

    print(ma_sample)  # numpy array

    ma_mod = ARIMA(ma_sample, order = (0, 0, 1)) # MA1 model
    #print(ar_mod)

    #ar_res = ar_mod.fit()
    #print(ar_res.summary())

    return ma_sample

generate_ar1(0.5, 10)
generate_ma1(0.5, 10)

# arparams = np.array([.75, -.25])
# maparams = np.array([.65, .35])
#
# arparams = np.r_[1, -arparams]
# maparams = np.r_[1, maparams]
# nobs = 250
# y = arma_generate_sample(arparams, maparams, nobs)
#
# dates = pd.date_range('1980-1-1', freq="M", periods=nobs)
# y = pd.Series(y, index=dates)
# arma_mod = ARIMA(y, order=(2, 0, 2), trend='n')
# arma_res = arma_mod.fit()
#
# print(arma_res.summary())
#
#
# fig, ax = plt.subplots(figsize=(10,8))
# fig = plot_predict(arma_res, start='1999-06-30', end='2001-05-31', ax=ax)
# legend = ax.legend(loc='upper left')
# plt.show()