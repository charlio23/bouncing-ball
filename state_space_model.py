from operator import matmul
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg
from tqdm import tqdm


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

def initialize_random(T, hid_dim, D):
    mu_0 = np.random.randn(hid_dim)*10
    P_0 = np.eye(hid_dim)*100
    A = np.random.randn(hid_dim,hid_dim)*1 + np.eye(hid_dim)*1
    cov_1 = np.eye(hid_dim,hid_dim)*1
    B =  np.eye(D, hid_dim)
    cov_2 = np.eye(D,D)*1
    return mu_0, P_0, A, cov_1, B, cov_2

def initialize_from_params(params):
    return params['mu_0'], params['P_0'], params['A'], params['cov_1'], params['B'], params['cov_2']

def filtering(mu_0, P_0, A, cov_1, B, cov_2, obs):
    T, D = obs.shape
    (hid_dim,) = mu_0.shape
    mu = np.zeros((T, hid_dim))
    V = np.zeros((T, hid_dim, hid_dim))
    P = np.zeros_like(V)
    for t in range(T):
        if t == 0:
            K = np.dot(np.dot(P_0,B.T), np.linalg.inv(np.dot(np.dot(B,P_0),B.T) + cov_2))
            mu[t,:] = mu_0 + np.dot(K, (obs[t,:] - np.dot(B, mu_0.reshape(hid_dim,1)).reshape(-1)).reshape(D,1)).reshape(-1)
            V[t,:,:] = np.dot(np.eye(hid_dim) - np.dot(K,B), P_0)
        else:
            K = np.dot(np.dot(P[t-1,:,:],B.T),np.linalg.pinv(np.dot(np.dot(B,P[t-1,:,:]),B.T) + cov_2))
            mu[t,:] = np.dot(A,mu[t-1,:].reshape(hid_dim,1)).reshape(-1) + np.dot(K, (obs[t,:] - np.dot(B, np.dot(A,mu[t-1,:].reshape(hid_dim,1))).reshape(-1)).reshape(D,1)).reshape(-1)
            V[t,:,:] = np.dot(np.eye(hid_dim) - np.dot(K,B), P[t,:,:])          

        P[t,:,:] = np.dot(np.dot(A,V[t,:,:]),A.T) + cov_1

    return mu, V, P


def smoothing(A, mu, V, P):
    T, hid_dim = mu.shape
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
            mu_smoothed[t,:] = mu[t,:] + np.dot(J[t,:,:], (mu_smoothed[t+1,:] - np.dot(A, mu[t,:].reshape(hid_dim,1)).reshape(-1)).reshape(hid_dim,1)).reshape(-1)
            V_smoothed[t,:,:] = V[t,:,:] + np.dot(np.dot(J[t,:,:], (V_smoothed[t+1,:,:] - P[t,:,:])),J[t,:,:].T)
    
    return mu_smoothed, V_smoothed, J

def expectation(mu_smoothed, V_smoothed, J):
    T, hid_dim = mu_smoothed.shape
    E_z = mu_smoothed
    E_z_z = V_smoothed + np.matmul(mu_smoothed.reshape(T,hid_dim,1), mu_smoothed.reshape(T,1,hid_dim))
    E_z_z_1 = np.matmul(V_smoothed[1:,:,:], J[:-1,:,:].transpose(0,2,1)) + np.matmul(mu_smoothed[1:,:].reshape(T-1,hid_dim,1), mu_smoothed[:-1,:].reshape(T-1,1,hid_dim))
    return E_z, E_z_z, E_z_z_1

def maximization(E_z, E_z_z, E_z_z_1,obs):
    T, hid_dim = E_z.shape
    _, D = obs.shape
    mu_0 = E_z[0]
    P_0 = E_z_z[0,:,:] - np.dot(E_z[0:1,:].T,E_z[0:1,:])

    A = np.dot(np.sum(E_z_z_1,axis=0), np.linalg.inv(np.sum(E_z_z[:-1,:,:], axis=0)))
    
    cov_1_term_1 = np.matmul(A.reshape(1,hid_dim,hid_dim),E_z_z_1.transpose(0,2,1))
    cov_1_term_2 = np.matmul(E_z_z_1, A.T.reshape(1,hid_dim,hid_dim))
    cov_1_term_3 = np.matmul(A.reshape(1,hid_dim,hid_dim),np.matmul(E_z_z[:-1,:,:], A.T.reshape(1,hid_dim,hid_dim)))
    cov_1 = (1/(T-1))*np.sum(E_z_z[1:,:,:] - cov_1_term_1 - cov_1_term_2 + cov_1_term_3,axis=0)

    B = np.dot(np.sum(np.matmul(obs.reshape(T,D,1), E_z.reshape(T,1,hid_dim)),axis=0), np.linalg.inv(np.sum(E_z_z, axis=0)))
    
    cov_2_term_1 = np.matmul(obs.reshape(T,D,1), obs.reshape(T,1,D))
    cov_2_term_2 = np.matmul(B.reshape(1,D,hid_dim),np.matmul(E_z.reshape(T,hid_dim,1), obs.reshape(T,1,D)))
    cov_2_term_3 = np.matmul(np.matmul(obs.reshape(T,D,1), E_z.reshape(T,1,hid_dim)), B.T.reshape(1,hid_dim,D))
    cov_2_term_4 = np.matmul(B.reshape(1,D,hid_dim),np.matmul(E_z_z, B.T.reshape(1,hid_dim,D)))
    cov_2 = (1/T)*np.sum(cov_2_term_1 - cov_2_term_2 - cov_2_term_3 + cov_2_term_4,axis=0)

    return mu_0, P_0, A, cov_1, B, cov_2

def loglikelihood(obs, mu_0, P_0, A, cov_1, B, cov_2, mu_smoothed, V_smoothed, J):
    T, hid_dim = mu_smoothed.shape
    D, _ = cov_2.shape
    #Calculate terms with initial latent variable
    E_0 = np.trace(np.dot(np.linalg.inv(P_0), V_smoothed[0,:,:])) 
    q_0 = -(1/2)*np.log(np.linalg.det(P_0))
    q_0 += -(1/2)*E_0
    
    #Calculate terms with transitioning latent variables
    inv_cov_1 = np.linalg.inv(cov_1)
    E_z = np.trace(np.matmul(inv_cov_1.reshape(1,hid_dim,hid_dim), V_smoothed[1:,:,:]), axis1=1, axis2=2)
    
    prod_A_cov = np.dot(A.T, np.dot(inv_cov_1, A))
    E_z += np.trace(np.matmul(prod_A_cov.reshape(1,hid_dim,hid_dim), V_smoothed[1:,:,:]), axis1=1, axis2=2)

    prod_A_J_cov = np.matmul(np.dot(A.T, inv_cov_1).reshape(1,hid_dim,hid_dim), J[1:,:,:])
    E_z += -np.trace(np.matmul(prod_A_J_cov, V_smoothed[1:,:,:]), axis1=1, axis2=2)

    prod_A_J_cov_2 = np.matmul(np.dot(inv_cov_1, A).reshape(1,hid_dim,hid_dim), V_smoothed[1:,:,:])
    E_z += -np.trace(np.matmul(prod_A_J_cov_2, J[1:,:,:].transpose(0,2,1)), axis1=1, axis2=2)

    dif_z = mu_smoothed[1:,:].reshape(T-1,hid_dim,1) - np.matmul(A.reshape(1,hid_dim,hid_dim), mu_smoothed[:-1,:].reshape(T-1,hid_dim,1))

    E_z += np.matmul(dif_z.transpose(0,2,1), np.matmul(inv_cov_1.reshape(1,hid_dim,hid_dim), dif_z)).reshape(T-1)

    q_z = -((T-1)/2)*np.log(np.linalg.det(cov_1))
    q_z += -(1/2)*np.sum(E_z, axis=0)

    #Calculate terms with observational distribution
    inv_cov_2 = np.linalg.inv(cov_2)
    prod_B_cov = np.dot(B.T, np.dot(inv_cov_2, B))
    E_x = np.trace(np.matmul(prod_B_cov.reshape(1,hid_dim,hid_dim), V_smoothed), axis1=1, axis2=2)

    dif_x = obs.reshape(T,D,1) - np.matmul(B.reshape(1,D,hid_dim), mu_smoothed.reshape(T,hid_dim,1))
    E_x += np.matmul(dif_x.transpose(0,2,1), np.matmul(inv_cov_2.reshape(1,D,D), dif_x)).reshape(T)
    q_x = -(T/2)*np.log(np.linalg.det(cov_2))
    q_x += -(1/2)*np.sum(E_x, axis=0)

    loglikeli = q_0 + q_z + q_x

    return loglikeli

def fit(obs, hid_dim, T, itMax=70, init_params=None):
    D = obs.shape[1]
    # Initialization
    if init_params is None:
        mu_0, P_0, A, cov_1, B, cov_2 = initialize_random(T, hid_dim, D)
    else:
        mu_0, P_0, A, cov_1, B, cov_2 = initialize_from_params(init_params)
    for i in range(itMax):
        # filtering
        mu, V, P = filtering(mu_0, P_0, A, cov_1, B, cov_2, obs[:T,:])
        # smoothing
        mu_smoothed, V_smoothed, J = smoothing(A, mu, V, P)
        # expectation
        E_z, E_z_z, E_z_z_1 = expectation(mu_smoothed, V_smoothed, J)
        # maximization
        mu_0, P_0, A, cov_1, B, cov_2 = maximization(E_z, E_z_z, E_z_z_1, obs[:T,:])

    loglike = loglikelihood(obs[:T,:], mu_0, P_0, A, cov_1, B, cov_2, mu_smoothed, V_smoothed, J)
    result = {
        'mu_0': mu_0,
        'P_0': P_0,
        'A': A,
        'B': B,
        'cov_1': cov_1,
        'cov_2': cov_2,
        'mu_smoothed': mu_smoothed
    }
    return result, loglike

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
# Finalization of seeding initialization process
print(best_init[1])
result = best_init[0]
print(result)
prediction = np.matmul(result['B'].reshape(1,D,hid_dim),result['mu_smoothed'].reshape(T,hid_dim,1))
plt.scatter(noisy_data[:T,0],noisy_data[:T,1])
plt.plot(data[:T,0],data[:T,1],label='true')
plt.plot(prediction[:,0],prediction[:,1],label='estimated')
plt.legend()
plt.show()
