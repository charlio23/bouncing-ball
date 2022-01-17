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
noisy_data = data[:,:2] + np.random.randn(T,D)*20
#noisy_data = (noisy_data - np.mean(noisy_data,axis=0))/np.std(noisy_data,axis=0)
# Linear Gaussian State-Space Model
"""
X_t = A X_t-1 + U_t
Y_t = BX_t + V_t
We will perform sequential inference using a Kalman filter
Parameters, mu_0, P_o, A, cov_1, B, cov_2
"""


def initialize_from_params(params):
    return params['mu_0'], params['P_0'], params['A'], params['cov_1'], params['B'], params['cov_2']

"""
# Dimensions setup
## Select length
hid_dim = 4
T = 100
best_init = None
converged_rate = 0
for i in tqdm(range(100)):
    try:
        result, loglike = fit(noisy_data, hid_dim, T)
        if not np.isnan(loglike):
            converged_rate += 1
            if best_init is None:
                best_init = (result, loglike)
            elif (loglike > best_init[1]):
                best_init = (result, loglike)
    except np.linalg.LinAlgError:
        continue
print(converged_rate*100/100)
"""

kf = KalmanFilter(hid_dim=2, obs_dim=2)
kf.set_input(noisy_data)
(filt_means, filt_cov) = kf.filter()
#print(filt_means)

from pykalman import KalmanFilter as KF2
kf2 = KF2(n_dim_state=2, n_dim_obs=2)
(filt_means2, filt_cov2) = kf2.filter(noisy_data)
#print(filt_means - filt_means2)
print(filt_cov[:3])
print(filt_cov2[:3])
exit()
kf.em(noisy_data, itMax=50)

kf.filter()
kf.smooth()

preds = np.dot( kf.C, kf.smoothed_state_mean.T).T

plt.scatter(noisy_data[:T,0],noisy_data[:T,1])
plt.plot(data[:T,0],data[:T,1],label='true')
plt.plot(preds[:,0],preds[:,1],label='estimated')
plt.legend()
plt.show()
