import numpy as np
from matplotlib import pyplot as plt

np.random.seed(42)

## Load data 
data = np.load('./datasets/trimmed_datapoints.npy')
N, T, D = data.shape
noisy_data = data + np.random.randn(N, T,D)*5
print(noisy_data.shape)
for i in range(N):
    plt.plot(noisy_data[i,:,0],noisy_data[i,:,1], color='b')
plt.show()

# Linear Gaussian State-Space Model
"""
X_t = A X_t-1 + U_t
Y_t = BX_t + V_t
We will perform sequential inference using a Kalman filter
Parameters, mu_0, P_o, A, cov_1, B, cov_2
"""

def initialize(T, hid_dim, D):
    mu_0 = np.random.randn(hid_dim)
    P_0 = np.eye(hid_dim)*0.01
    A = np.random.randn(hid_dim,hid_dim)*0.01
    cov_1 = np.random.randn(hid_dim,hid_dim)*0.01
    B = np.eye(D, hid_dim)
    cov_2 = np.random.randn(D,D)*0.01
    return mu_0, P_0, A, cov_1, B, cov_2

def filtering(mu_0, P_0, A, cov_1, B, cov_2, obs):
    N, T, D = obs.shape
    (hid_dim,) = mu_0.shape
    mu = np.zeros((N, T, hid_dim))
    V = np.zeros((T, hid_dim, hid_dim))
    P = np.zeros_like(V)
    for t in range(T):
        if t == 0:
            K = np.dot(np.dot(P_0,B.T), np.linalg.inv(np.dot(np.dot(B,P_0),B.T) + cov_2))
            mu_prod_term = (obs[:,t,:] - np.dot(B, mu_0.reshape(hid_dim,1)).reshape(1,-1)).reshape(N,D,1)
            mu[:,t,:] = mu_0 + np.matmul(K.reshape(1,hid_dim,D), mu_prod_term).reshape(N,-1)
            V[t,:,:] = np.matmul(np.eye(hid_dim) - np.dot(K,B), P_0)
        else:
            K = np.dot(np.dot(P[t-1,:,:],B.T),np.linalg.inv(np.dot(np.dot(B,P[t-1,:,:]),B.T) + cov_2))
            mu[:,t,:] = np.matmul(A.reshape(1,hid_dim,hid_dim),mu[:,t-1,:].reshape(N,hid_dim,1)).reshape(N,-1) + np.matmul(K.reshape(1,hid_dim,D), (obs[:,t,:] - np.matmul(B.reshape(1,D,hid_dim), np.matmul(A.reshape(1,hid_dim,hid_dim),mu[:,t-1,:].reshape(N,hid_dim,1))).reshape(N,-1)).reshape(N,D,1)).reshape(N,-1)
            V[t,:,:] = np.dot(np.eye(hid_dim) - np.dot(K,B), P[t,:,:])          

        P[t,:,:] = np.dot(np.dot(A,V[t,:,:]),A.T) + cov_1

    return mu, V, P


def smoothing(A, mu, V, P):
    N, T, hid_dim = mu.shape
    mu_smoothed = np.zeros_like(mu)
    V_smoothed = np.zeros_like(V)
    J = np.zeros_like(V)
    for i in range(T):
        t = T - 1 - i
        J[t,:,:] = np.dot(np.dot(V[t,:,:], A.T), np.linalg.inv(P[t,:,:]))
        if t == (T-1):
            mu_smoothed[:,t,:] = mu[:,t,:]
            V_smoothed[t,:,:] = V[t,:,:]
        else:
            mu_prod_term =  (mu_smoothed[:,t+1,:] - np.matmul(A.reshape(1,hid_dim,hid_dim), mu[:,t,:].reshape(N,hid_dim,1)).reshape(N,-1)).reshape(N,hid_dim,1)
            mu_smoothed[:,t,:] = mu[:,t,:] + np.matmul(J[t,:,:].reshape(1,hid_dim,hid_dim), mu_prod_term).reshape(N,-1)
            V_smoothed[t,:,:] = V[t,:,:] + np.dot(np.dot(J[t,:,:], (V_smoothed[t+1,:,:] - P[t,:,:])),J[t,:,:].T)
    
    return mu_smoothed, V_smoothed, J

def expectation(mu_smoothed, V_smoothed, J):
    N, T, hid_dim = mu_smoothed.shape
    E_z = mu_smoothed
    E_z_z = V_smoothed.reshape(1,T,hid_dim,hid_dim) + np.matmul(mu_smoothed.reshape(N, T,hid_dim,1), mu_smoothed.reshape(N,T,1,hid_dim))
    v_j_prod = np.matmul(V_smoothed[1:,:,:], J[:-1,:,:].transpose(0,2,1)).reshape(1,T-1,hid_dim,hid_dim)
    mu_mu_1 = np.matmul(mu_smoothed[:,1:,:].reshape(N,T-1,hid_dim,1), mu_smoothed[:,:-1,:].reshape(N,T-1,1,hid_dim))
    E_z_z_1 = v_j_prod + mu_mu_1
    return E_z, E_z_z, E_z_z_1

def maximization(E_z, E_z_z, E_z_z_1,obs):

    N, T, hid_dim = E_z.shape
    _, _, D = obs.shape
    mu_0 = (1/N)*np.sum(E_z[:,0,:],axis=0)
    P_0 = np.sum(E_z_z[:,0,:,:],axis=0) - np.matmul(np.sum(E_z[:,0:1,:],axis=0).T,np.sum(E_z[:,0:1,:],axis=0))
    P_0 = (1/N)*P_0

    A = np.dot(np.sum(E_z_z_1,axis=(0,1)), np.linalg.inv(np.sum(E_z_z[:,:-1,:,:], axis=(0,1))))

    cov_1_term_1 = np.matmul(A.reshape(1,1,hid_dim,hid_dim),E_z_z_1.transpose(0,1,3,2))
    cov_1_term_2 = np.matmul(E_z_z_1, A.T.reshape(1,1,hid_dim,hid_dim))
    cov_1_term_3 = np.matmul(A.reshape(1,1,hid_dim,hid_dim),np.matmul(E_z_z[:,:-1,:,:], A.T.reshape(1,1,hid_dim,hid_dim)))
    cov_1 = (1/(N*(T-1)))*np.sum(E_z_z[:,1:,:,:] - cov_1_term_1 - cov_1_term_2 + cov_1_term_3,axis=(0,1))

    B = np.dot(np.sum(np.matmul(obs.reshape(N,T,D,1), E_z.reshape(N,T,1,hid_dim)),axis=(0,1)), np.linalg.inv(np.sum(E_z_z, axis=(0,1))))
    
    cov_2_term_1 = np.matmul(obs.reshape(N,T,D,1), obs.reshape(N,T,1,D))
    cov_2_term_2 = np.matmul(B.reshape(1,D,hid_dim),np.matmul(E_z.reshape(N,T,hid_dim,1), obs.reshape(N,T,1,D)))
    cov_2_term_3 = np.matmul(np.matmul(obs.reshape(N,T,D,1), E_z.reshape(N,T,1,hid_dim)), B.T.reshape(1,1,hid_dim,D))
    cov_2_term_4 = np.matmul(B.reshape(1,1,D,hid_dim),np.matmul(E_z_z, B.T.reshape(1,1,hid_dim,D)))
    cov_2 = (1/(T*N))*np.sum(cov_2_term_1 - cov_2_term_2 - cov_2_term_3 + cov_2_term_4,axis=(0,1))

    return mu_0, P_0, A, cov_1, B, cov_2



# Dimensions setup
## Select length
hid_dim = 4
N, T, D = noisy_data.shape
# Initialization
mu_0, P_0, A, cov_1, B, cov_2 = initialize(T, hid_dim, D)
print(noisy_data[0])
for i in range(100):
    # filtering
    mu, V, P = filtering(mu_0, P_0, A, cov_1, B, cov_2, noisy_data)
    # smoothing
    mu_smoothed, V_smoothed, J = smoothing(A, mu, V, P)
    # expectation
    E_z, E_z_z, E_z_z_1 = expectation(mu_smoothed, V_smoothed, J)
    # maximization
    mu_0, P_0, A, cov_1, B, cov_2 = maximization(E_z, E_z_z, E_z_z_1, noisy_data)


# Convergence of the algorithm
print("Init:")
print(mu_0)
print(P_0)
print("X:")
print(A)
print(cov_1)

print("Y:")
print(B)
print(cov_2)


y = np.matmul(B.reshape(1,1,D,hid_dim),mu_smoothed.reshape(N,T,hid_dim,1)).reshape(N,T,D)

print((np.square(y - noisy_data)).mean(axis=(0,1)))

for i in range(N):
    plt.plot(noisy_data[i,:,0],noisy_data[i,:,1], color='b')
    plt.plot(y[i,:,0],y[i,:,1], color='g')
plt.show()

