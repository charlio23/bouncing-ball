import numpy as np
from matplotlib import pyplot as plt


## Load data 
data = np.load('./datasets/positions.npy')
T, D = data.shape

data_trimmed = []
pre_iter = 1
for t in range(2,T):
    vel = np.abs(data[t,2] - data[t-1,2]) + np.abs(data[t,3] - data[t-1,3])
    
    if vel > 0.1:
        dist = t - pre_iter
        data_slice = data[pre_iter+1:pre_iter+dist]
        pre_iter = t
        if dist > 30:
            data_trimmed.append(data_slice[:30,:2])
data_trimmed = np.array(data_trimmed)
print(data_trimmed.shape)
np.save('trimmed_datapoints.npy',data_trimmed)
