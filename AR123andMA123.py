import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample
import statsmodels
import statsmodels.api as sm

np.random.seed(1)

def AR1(timestep):
  rho = np.random.uniform(-1,1,1)+1e-6
  rho=float(rho)
  ar_coefs = [1,-rho]
  ma_coefs = [1,0]
  ar1=arma_generate_sample(ar_coefs, ma_coefs, nsample=timestep, sigma=1)
  ar1=np.array(ar1)
  return ar1

def AR2(timestep):
  while True:
    rho2 = np.random.uniform(-1,1,1)+1e-6
    rho1 = np.random.uniform(-3,3,1)+1e-6
    rho1=float(rho1)
    rho2=float(rho2)
    ar_coefs = [1,-rho1,-rho2]
    ma_coefs = [1,0]
    ar2_coef = sm.tsa.ArmaProcess(ar_coefs, ma_coefs)
    stationary=ar2_coef.isstationary
    if stationary ==False:
      continue
    else:
      ar2=arma_generate_sample(ar_coefs, ma_coefs, nsample=timestep, sigma=1)
      ar2=np.array(ar2)
      return ar2

def AR3(timestep):
  while True:
    rho1,rho2,rho3 = np.random.uniform(-3,3,3)+1e-6
    rho1=float(rho1)
    rho2=float(rho2)
    rho3=float(rho3)
    ar_coefs = [1,-rho1,-rho2,-rho3] #for rho should be inversed
    ma_coefs = [1,0]
    ar3_coef = sm.tsa.ArmaProcess(ar_coefs, ma_coefs)
    stationary=ar3_coef.isstationary
    if stationary==False:
      continue
    else:
      ar3=arma_generate_sample(ar_coefs, ma_coefs, nsample=timestep, sigma=1)
      ar3=np.array(ar3)
      return ar3
#stationary if -rho3*B^3-rho2*B^2-rho1*B+1=0 has roots outside unitcircle


def MA1(timestep):
  theta = np.random.uniform(-1,1,1)+1e-6
  theta=float(theta)
  ar_coefs = [1,0]
  ma_coefs = [1,theta]
  ma1=arma_generate_sample(ar_coefs, ma_coefs, nsample=timestep, sigma=1)
  ma1=np.array(ma1)
  return ma1


def MA2(timestep):
  while True:
    theta2 = np.random.uniform(-1,1,1)+1e-6
    theta1 = np.random.uniform(-3,3,1)+1e-6
    theta1=float(theta1)
    theta2=float(theta2)
    ar_coefs = [1,0]
    ma_coefs = [1,theta1,theta2]
    ma2_coef = sm.tsa.ArmaProcess(ar_coefs, ma_coefs)
    invertibility=ma2_coef.isinvertible
    if invertibility==False:
      continue
    else:
      ma2=arma_generate_sample(ar_coefs, ma_coefs, nsample=timestep, sigma=1)
      ma2=np.array(ma2)
      return ma2

      # -1<theta2<1, theta2+theta1>-1, theta1-theta2<1


def MA3(timestep):
  while True:
    theta1,theta2,theta3 = np.random.uniform(-3,3,3)+1e-6
    theta1=float(theta1)
    theta2=float(theta2)
    theta3=float(theta3)
    ar_coefs = [1,0]
    ma_coefs = [1,theta1,theta2,theta3]
    ma3_coef = sm.tsa.ArmaProcess(ar_coefs, ma_coefs)
    invertibility=ma3_coef.isinvertible
    if invertibility ==False:
      continue
    else:
      ma3=arma_generate_sample(ar_coefs, ma_coefs, nsample=timestep, sigma=1)
      ma3=np.array(ma3)
      return ma3

def scale(X,con):
  X=X/max(abs(X))
  X=X[con:]
  return X

def lstm(X,rs,nsample):
  X=X.reshape(rs,nsample,1)
  return X