import numpy as np
from matplotlib import pyplot as plt
from numpy.core.numeric import zeros_like
from pygame.constants import JOYAXISMOTION


## Load data 
data = np.load('./datasets/positions.npy')
T, D = data.shape
noisy_data = data + np.random.randn(T,D)*1

noisy_data = (noisy_data - np.mean(noisy_data,axis=0))/np.std(noisy_data,axis=0)

plt.plot(noisy_data[:20,0],noisy_data[:20,1])
plt.show()
# Linear Gaussian State-Space Model
"""
X_t = A X_t-1 + U_t
Y_t = BX_t + V_t
We will perform sequential inference using a Kalman filter
Parameters, mu_0, P_o, A, cov_1, B, cov_2
"""

def initialize(T, D_1, D_2):
    mu_0 = np.random.randn(D_1)
    P_0 = np.random.randn(D_1,D_1)
    A = np.random.randn(D_1,D_1)
    cov_1 = np.random.randn(D_1,D_1)
    B = np.random.randn(D_2,D_1)
    cov_2 = np.random.randn(D_2,D_2)
    return mu_0, P_0, A, cov_1, B, cov_2

def filtering(mu_0, P_0, A, cov_1, B, cov_2, obs):
    T, D = obs.shape
    mu = np.zeros((T, D))
    V = np.zeros((T, D, D))
    P = np.zeros_like(V)
    for t in range(T):
        if t == 0:
            K = np.dot(np.dot(P_0,B.T), np.linalg.inv(np.dot(np.dot(B,P_0),B.T) + cov_2))
            mu[t,:] = mu_0 + np.dot(K, (obs[t,:] + np.dot(B, mu_0.reshape(D,1)).reshape(-1)))
            V[t,:,:] = np.dot(np.eye(D) - np.dot(K,B), P_0)
        else:
            K = np.dot(np.dot(P[t-1,:,:],B.T),np.linalg.inv(np.dot(np.dot(B,P[t-1,:,:]),B.T) + cov_2))
            mu[t,:] = np.dot(A,mu[t-1,:].reshape(D,1)).reshape(-1) + np.dot(K, (obs[t,:] + np.dot(B, np.dot(A,mu[t-1,:].reshape(D,1))).reshape(-1)))
            V[t,:,:] = np.dot(np.eye(D) - np.dot(K,B), P[t,:,:])          

        P[t,:,:] = np.dot(np.dot(A,V[t,:,:]),A.T) + cov_1

    return mu, V, P


def smoothing(A, mu, V, P):
    T, D = mu.shape
    mu_smoothed = np.zeros_like(mu)
    V_smoothed = np.zeros_like(V)
    J = np.zeros_like(V)
    for i in range(T):
        t = T - 1 - i
        J[t,:,:] = np.dot(np.dot(V[t,:,:], A.T), np.linalg.inv(P[t,:,:]))
        if t == (T-1):
            mu_smoothed[t,:] = mu[t,:]
            V_smoothed[t,:,:] = V[t,:,:]
        else:
            mu_smoothed[t,:] = mu[t,:] + np.dot(J[t,:,:], (mu_smoothed[t+1,:] - np.dot(A, mu[t,:].reshape(D,1)).reshape(-1)))
            V_smoothed[t,:,:] = V[t,:,:] + np.dot(np.dot(J[t,:,:], (V_smoothed[t+1,:] - P[t,:,:])),J[t,:,:].T)
    return mu_smoothed, V_smoothed, J

def expectation(mu_smoothed, V_smoothed, J):
    T, D = mu_smoothed.shape
    E_z = mu_smoothed
    E_z_z = V_smoothed + np.matmul(mu_smoothed.reshape(T,D,1), mu_smoothed.reshape(T,1,D))
    E_z_z_1 = np.matmul(V_smoothed[1:,:,:], J[:-1,:,:].transpose(0,2,1)) + np.matmul(mu_smoothed[1:,:].reshape(T-1,D,1), mu_smoothed[:-1,:].reshape(T-1,1,D))
    return E_z, E_z_z, E_z_z_1

def maximization(E_z, E_z_z, E_z_z_1,obs):
    T, D = E_z.shape
    mu_0 = E_z[0]
    P_0 = E_z_z[0] - np.dot(E_z[0:1,:].T,E_z[0:1,:])

    A = np.dot(np.sum(E_z_z_1,axis=0), np.linalg.inv(np.sum(E_z_z[:-1,:,:], axis=0)))
    cov_1 = np.zeros_like(A)
    cov_1_term_1 = np.matmul(A.reshape(1,D,D),E_z_z_1.transpose(0,1,2))
    cov_1_term_2 = np.matmul(E_z_z_1, A.T.reshape(1,D,D))
    cov_1_term_3 = np.matmul(A.reshape(1,D,D),np.matmul(E_z_z[:-1,:,:], A.T.reshape(1,D,D)))
    cov_1 = (1/(T-1))*np.sum(E_z_z[1:,:,:] - cov_1_term_1 - cov_1_term_2 + cov_1_term_3,axis=0)

    B = np.dot(np.sum(np.matmul(obs.reshape(T,D,1), E_z.reshape(T,1,D)),axis=0), np.linalg.inv(np.sum(E_z_z, axis=0)))
    cov_2 = np.zeros_like(B)
    cov_2_term_1 = np.matmul(obs.reshape(T,D,1), obs.reshape(T,1,D))
    cov_2_term_2 = np.matmul(B.T.reshape(1,D,D),np.matmul(E_z.reshape(T,D,1), obs.reshape(T,1,D)))
    cov_2_term_3 = np.matmul(np.matmul(obs.reshape(T,D,1), E_z.reshape(T,1,D)), B.reshape(1,D,D))
    cov_2_term_4 = np.matmul(B.T.reshape(1,D,D),np.matmul(E_z_z, B.reshape(1,D,D)))
    cov_2 = (1/T)*np.sum(cov_2_term_1 - cov_2_term_2 - cov_2_term_3 + cov_2_term_4,axis=0)

    return mu_0, P_0, A, cov_1, B, cov_2



# Dimensions setup
## Select length
T = 20
D_1 = D_2 = noisy_data.shape[1]
# Initialization
mu_0, P_0, A, cov_1, B, cov_2 = initialize(T, D_1, D_2)

for i in range(100):
    # filtering
    mu, V, P = filtering(mu_0, P_0, A, cov_1, B, cov_2, noisy_data[:T,:])
    # smoothing
    mu_smoothed, V_smoothed, J = smoothing(A, mu, V, P)
    # expectation
    E_z, E_z_z, E_z_z_1 = expectation(mu_smoothed, V_smoothed, J)
    # maximization
    mu_0, P_0, A, cov_1, B, cov_2 = maximization(E_z, E_z_z, E_z_z_1, noisy_data[:T,:])
    print("X:")
    print(A)
    print(cov_1)

    print("Y:")
    print(B)
    print(cov_2)


# Convergence of the algorithm
print("X:")
print(A)
print(cov_1)

print("Y:")
print(B)
print(cov_2)