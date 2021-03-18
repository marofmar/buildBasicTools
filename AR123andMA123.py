import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample
import statsmodels
import statsmodels.api as sm

np.random.seed(1)

def AR1(timestep):  # timestep refers to the number of sequence elements
  """
  <AR for AutoRegressive>
  :param timestep:
  :return:
  """
  rho = np.random.rand((-1,1),1).round(3)  # from uniform distribution of (-1,1), pick 1 and round to the 3rd decimal
  ar_coefs = [1,-rho]  # following the old tradition(convention) of signal processing, opposite sign (+ to -)
  ma_coefs = [1,0]  # 0 is for the zero-lag
  ar1 = arma_generate_sample(ar_coefs, ma_coefs, nsample=timestep, sigma=1)  # sigma: std dev of noise
  ar1 = np.array(ar1)
  return ar1

def AR2(timestep):
  """
  https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_process.ArmaProcess.html
  :param timestep: the number of sequence (one at each)
  :return: AR2 numpy array
  """
  while True:
    rho2 = np.random.rand((-1,1), 1) # from random uniform range(-1, 1)
    rho2 = np.random.rand((-3,3), 1) # from random uniform range(-3, 3)
    ar_coefs = [1,-rho1,-rho2] # 1 is for zero-lag
    ma_coefs = [1,0]
    ar2_coef = sm.tsa.ArmaProcess(ar_coefs, ma_coefs)
    stationary=ar2_coef.isstationary
    if stationary ==False:  # if not stationary go up to the loop and find another rho
      continue
    else: # if it is stationary, generate designated number of sequential elements
      ar2 = arma_generate_sample(ar_coefs, ma_coefs, nsample=timestep, sigma=1)
      ar2 = np.array(ar2)
      return ar2

def AR3(timestep):
  while True:
    rho1, rho2, rho3 = np.random.rand((-3,3), 3) # select three rhos for the AR
    ar_coefs = [1,-rho1,-rho2,-rho3]
    ma_coefs = [1,0]
    ar3_coef = sm.tsa.ArmaProcess(ar_coefs, ma_coefs)
    stationary=ar3_coef.isstationary
    if stationary==False:
      continue
    else:
      ar3=arma_generate_sample(ar_coefs, ma_coefs, nsample=timestep, sigma=1)
      return ar3
#stationary if -rho3*B^3-rho2*B^2-rho1*B+1=0 has roots outside unitcircle


def MA1(timestep):
  """
  <MA for Moving-Average>
  :param timestep:
  :return:
  """
  theta = np.random.rand((-1,1), 1)
  ar_coefs = [1,0]
  ma_coefs = [1,theta]
  ma1 = arma_generate_sample(ar_coefs, ma_coefs, nsample=timestep, sigma=1)
  return ma1


def MA2(timestep):
  while True:
    theta2 = np.random.rand((-1,1), 1)
    theta1 = np.random.rand((-3, 3), 1)
    ar_coefs = [1, 0]
    ma_coefs = [1, theta1, theta2]
    ma2_coef = sm.tsa.ArmaProcess(ar_coefs, ma_coefs)
    invertibility = ma2_coef.isinvertible
    if invertibility == False:
      continue
    else:
      ma2 = arma_generate_sample(ar_coefs, ma_coefs, nsample=timestep, sigma=1)
      ma2 = np.array(ma2)
      return ma2

      # -1<theta2<1, theta2+theta1>-1, theta1-theta2<1


def MA3(timestep):
  while True:
    theta1, theta2, theta3 = np.random.rand((-3, 3), 3)
    ar_coefs = [1, 0]
    ma_coefs = [1, theta1, theta2, theta3]
    ma3_coef = sm.tsa.ArmaProcess(ar_coefs, ma_coefs)
    invertibility = ma3_coef.isinvertible
    if invertibility ==False:
      continue
    else:
      ma3 = arma_generate_sample(ar_coefs, ma_coefs, nsample=timestep, sigma=1)
      ma3 = np.array(ma3)
      return ma3

def scale(X,con):
  """
  SCALER
  :param X:
  :param con:
  :return:
  """
  X = X/max(abs(X))
  X = X[con:]
  return X

def lstm(X, rs, nsample):
  X = X.reshape(rs,nsample,1)
  return X