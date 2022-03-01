from operator import matmul
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg
from tqdm import tqdm
from models.KalmanFilter import KalmanFilter


np.random.seed(10)
## Load data 
data = np.load('./datasets/kalman_positions.npy')
T, D = data[:,:2].shape
noisy_data = data[:,:2] + np.random.randn(T,D)*100
#noisy_data = (noisy_data - np.mean(noisy_data,axis=0))/np.std(noisy_data,axis=0)
# Linear Gaussian State-Space Model
"""
X_t = A X_t-1 + U_t
Y_t = BX_t + V_t
We will perform sequential inference using a Kalman filter
Parameters, mu_0, P_o, A, cov_1, B, cov_2
"""


kf = KalmanFilter(hid_dim=2, obs_dim=2)

kf.em(noisy_data, itMax=50)

kf.filter()
kf.smooth()

print('Likelihood is', kf.compute_loglikelihood())

preds = np.dot(kf.C, kf.smoothed_state_mean.T).T

plt.scatter(noisy_data[:T,0],noisy_data[:T,1])
plt.plot(data[:T,0],data[:T,1],label='true')
plt.plot(preds[:,0],preds[:,1],label='estimated')
plt.legend()
plt.show()

