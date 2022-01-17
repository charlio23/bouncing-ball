from operator import matmul
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg
from tqdm import tqdm
from pykalman import KalmanFilter


np.random.seed(10)
## Load data 
data = np.load('./datasets/kalman_positions.npy')
T, D = data[:,:2].shape
noisy_data = data[:,:2] + np.random.randn(T,D)*20
#noisy_data = (noisy_data - np.mean(noisy_data,axis=0))/np.std(noisy_data,axis=0)
# Linear Gaussian State-Space Model

print(data.shape)

kf = KalmanFilter(n_dim_state=2, n_dim_obs=2)

(filt_means, filt_cov) = kf.filter(noisy_data)

print(filt_means)

#print(filt_cov)

exit()

kf = kf.em(noisy_data, n_iter=50, em_vars='all')

(filt_means, filt_cov) = kf.filter(noisy_data)
(smooth_means, smooth_cov) = kf.smooth(noisy_data)

print(kf.transition_matrices)
print(kf.observation_matrices)
print(kf.transition_offsets)
print(kf.transition_covariance)
print(kf.observation_covariance)
print(kf.observation_offsets)

preds = (np.dot(kf.observation_matrices, smooth_means.T).T + kf.observation_offsets)

plt.scatter(noisy_data[:T,0],noisy_data[:T,1])
plt.plot(data[:T,0],data[:T,1],label='true')
plt.plot(preds[:,0], preds[:,1], label='pred')
#plt.plot(prediction[:,0],prediction[:,1],label='estimated')
plt.legend()
plt.show()