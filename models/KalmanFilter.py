import numpy as np

class KalmanFilter():

    def __init__(self, hid_dim, obs_dim):

        self.hid_dim = hid_dim
        self.obs_dim = obs_dim

        # Observation info
        self.observations = None
        self.T = None

        # State info
        self.filtered_state_mean = None
        self.filtered_state_cov = None
        self.smoothed_state_mean = None
        self.smoothed_state_cov = None

        self.P = None

        # Kalman filter parameters
        self.mu_0 = None
        self.P_0 = None
        self.A = None
        self.cov_1 = None
        self.C = None
        self.cov_2 = None

        # EM internal parameters
        self._E_z = None
        self._E_z_z = None
        self._E_z_z_1 = None

        # Log-likelihood
        self.loglikelihood = None

        self._initialize_params()

    def _initialize_params(self):
        
        self.mu_0 = np.random.randn(self.hid_dim)*10
        self.P_0 = np.eye(self.hid_dim)*100
        self.A = np.random.randn(self.hid_dim,self.hid_dim)*1 + np.eye(self.hid_dim)*1
        self.cov_1 = np.eye(self.hid_dim,self.hid_dim)*1
        self.C =  np.eye(self.obs_dim, self.hid_dim)
        self.cov_2 = np.eye(self.obs_dim,self.obs_dim)*1

        self.mu_0 = np.zeros(self.hid_dim)
        self.P_0 = np.eye(self.hid_dim)
        self.A = np.eye(self.hid_dim)
        self.cov_1 = np.eye(self.hid_dim,self.hid_dim)
        self.C =  np.eye(self.obs_dim, self.hid_dim)
        self.cov_2 = np.eye(self.obs_dim,self.obs_dim)

    def _expectation(self):
        self._E_z = self.smoothed_state_mean
        self._E_z_z = self.smoothed_state_cov + np.matmul(self.smoothed_state_mean.reshape(self.T,self.hid_dim,1), self.smoothed_state_mean.reshape(self.T,1,self.hid_dim))
        self._E_z_z_1 = np.matmul(self.smoothed_state_cov[1:,:,:], self.J[:-1,:,:].transpose(0,2,1)) + np.matmul(self.smoothed_state_mean[1:,:].reshape(self.T-1,self.hid_dim,1), self.smoothed_state_mean[:-1,:].reshape(self.T-1,1,self.hid_dim))

    def _maximization(self):
        self.mu_0 = self._E_z[0]
        self.P_0 = self._E_z_z[0,:,:] - np.dot(self._E_z[0:1,:].T, self._E_z[0:1,:])

        self.A = np.dot(np.sum(self._E_z_z_1,axis=0), np.linalg.pinv(np.sum(self._E_z_z[:-1,:,:], axis=0)))
        
        cov_1_term_1 = np.matmul(self.A.reshape(1,self.hid_dim,self.hid_dim),self._E_z_z_1.transpose(0,2,1))
        cov_1_term_2 = np.matmul(self._E_z_z_1, self.A.T.reshape(1,self.hid_dim,self.hid_dim))
        cov_1_term_3 = np.matmul(self.A.reshape(1,self.hid_dim,self.hid_dim),np.matmul(self._E_z_z[:-1,:,:], self.A.T.reshape(1,self.hid_dim,self.hid_dim)))
        self.cov_1 = (1/(self.T-1))*np.sum(self._E_z_z[1:,:,:] - cov_1_term_1 - cov_1_term_2 + cov_1_term_3,axis=0)

        self.C = np.dot(np.sum(np.matmul(self.observations.reshape(self.T,self.obs_dim,1), self._E_z.reshape(self.T,1,self.hid_dim)),axis=0), np.linalg.pinv(np.sum(self._E_z_z, axis=0)))
        
        cov_2_term_1 = np.matmul(self.observations.reshape(self.T,self.obs_dim,1), self.observations.reshape(self.T,1,self.obs_dim))
        cov_2_term_2 = np.matmul(self.C.reshape(1,self.obs_dim,self.hid_dim),np.matmul(self._E_z.reshape(self.T,self.hid_dim,1), self.observations.reshape(self.T,1,self.obs_dim)))
        cov_2_term_3 = np.matmul(np.matmul(self.observations.reshape(self.T,self.obs_dim,1), self._E_z.reshape(self.T,1,self.hid_dim)), self.C.T.reshape(1,self.hid_dim,self.obs_dim))
        cov_2_term_4 = np.matmul(self.C.reshape(1,self.obs_dim,self.hid_dim),np.matmul(self._E_z_z, self.C.T.reshape(1,self.hid_dim,self.obs_dim)))
        self.cov_2 = (1/self.T)*np.sum(cov_2_term_1 - cov_2_term_2 - cov_2_term_3 + cov_2_term_4,axis=0)

    def set_input(self, obs):
        self.T = obs.shape[0]
        self.observations = obs

    def filter(self):
        mu = np.zeros((self.T, self.hid_dim))
        V = np.zeros((self.T, self.hid_dim, self.hid_dim))
        self.P = np.zeros_like(V)
        for t in range(self.T):
            if t == 0:
                K = np.dot(np.dot(self.P_0,self.C.T), np.linalg.pinv(np.dot(np.dot(self.C,self.P_0),self.C.T) + self.cov_2))
                mu[t,:] = self.mu_0 + np.dot(K, (self.observations[t,:] - np.dot(self.C, self.mu_0.reshape(self.hid_dim,1)).reshape(-1)).reshape(self.obs_dim,1)).reshape(-1)
                V[t,:,:] = np.dot(np.eye(self.hid_dim) - np.dot(K,self.C), self.P_0)
            else:
                K = np.dot(np.dot(self.P[t-1,:,:],self.C.T),np.linalg.pinv(np.dot(np.dot(self.C,self.P[t-1,:,:]),self.C.T) + self.cov_2))
                mu[t,:] = np.dot(self.A,mu[t-1,:].reshape(self.hid_dim,1)).reshape(-1) + np.dot(K, (self.observations[t,:] - np.dot(self.C, np.dot(self.A,mu[t-1,:].reshape(self.hid_dim,1))).reshape(-1)).reshape(self.obs_dim,1)).reshape(-1)
                V[t,:,:] = np.dot(np.eye(self.hid_dim) - np.dot(K,self.C), self.P[t,:,:])          

            self.P[t,:,:] = np.dot(np.dot(self.A,V[t,:,:]),self.A.T) + self.cov_1

        self.filtered_state_mean = mu
        self.filtered_state_cov = V
        return (self.filtered_state_mean, self.filtered_state_cov)

    def smooth(self):
        mu_smoothed = np.zeros_like(self.filtered_state_mean)
        V_smoothed = np.zeros_like(self.filtered_state_cov)
        self.J = np.zeros_like(self.filtered_state_cov)
        for i in range(self.T):
            t = self.T - 1 - i
            self.J[t,:,:] = np.dot(np.dot(self.filtered_state_cov[t,:,:], self.A.T), np.linalg.pinv(self.P[t,:,:]))
            if t == (self.T-1):
                mu_smoothed[t,:] = self.filtered_state_mean[t,:]
                V_smoothed[t,:,:] = self.filtered_state_cov[t,:,:]
            else:
                mu_smoothed[t,:] = self.filtered_state_mean[t,:] + np.dot(self.J[t,:,:], (mu_smoothed[t+1,:] - np.dot(self.A, self.filtered_state_mean[t,:].reshape(self.hid_dim,1)).reshape(-1)).reshape(self.hid_dim,1)).reshape(-1)
                V_smoothed[t,:,:] = self.filtered_state_cov[t,:,:] + np.dot(np.dot(self.J[t,:,:], (V_smoothed[t+1,:,:] - self.P[t,:,:])),self.J[t,:,:].T)
        self.smoothed_state_mean = mu_smoothed
        self.smoothed_state_cov = V_smoothed
        return (self.smoothed_state_mean, self.smoothed_state_cov)

    def compute_loglikelihood(self):

        #Calculate terms with initial latent variable
        E_0 = np.trace(np.dot(np.linalg.pinv(self.P_0), self.smoothed_state_cov[0,:,:])) 
        q_0 = -(1/2)*np.log(np.linalg.det(self.P_0))
        q_0 += -(1/2)*E_0
        
        #Calculate terms with transitioning latent variables
        inv_cov_1 = np.linalg.pinv(self.cov_1)
        E_z = np.trace(np.matmul(inv_cov_1.reshape(1,self.hid_dim,self.hid_dim), self.smoothed_state_cov[1:,:,:]), axis1=1, axis2=2)
        
        prod_A_cov = np.dot(self.A.T, np.dot(inv_cov_1, self.A))
        E_z += np.trace(np.matmul(prod_A_cov.reshape(1,self.hid_dim,self.hid_dim), self.smoothed_state_cov[1:,:,:]), axis1=1, axis2=2)

        prod_A_J_cov = np.matmul(np.dot(self.A.T, inv_cov_1).reshape(1,self.hid_dim,self.hid_dim), self.J[1:,:,:])
        E_z += -np.trace(np.matmul(prod_A_J_cov, self.smoothed_state_cov[1:,:,:]), axis1=1, axis2=2)

        prod_A_J_cov_2 = np.matmul(np.dot(inv_cov_1, self.A).reshape(1,self.hid_dim,self.hid_dim), self.smoothed_state_cov[1:,:,:])
        E_z += -np.trace(np.matmul(prod_A_J_cov_2, self.J[1:,:,:].transpose(0,2,1)), axis1=1, axis2=2)

        dif_z = self.smoothed_state_mean[1:,:].reshape(self.T-1,self.hid_dim,1) - np.matmul(self.A.reshape(1,self.hid_dim,self.hid_dim), self.smoothed_state_mean[:-1,:].reshape(self.T-1,self.hid_dim,1))

        E_z += np.matmul(dif_z.transpose(0,2,1), np.matmul(inv_cov_1.reshape(1,self.hid_dim,self.hid_dim), dif_z)).reshape(self.T-1)

        q_z = -((self.T-1)/2)*np.log(np.linalg.det(self.cov_1))
        q_z += -(1/2)*np.sum(E_z, axis=0)

        #Calculate terms with observational distribution
        inv_cov_2 = np.linalg.pinv(self.cov_2)
        prod_B_cov = np.dot(self.C.T, np.dot(inv_cov_2, self.C))
        E_x = np.trace(np.matmul(prod_B_cov.reshape(1,self.hid_dim,self.hid_dim), self.smoothed_state_cov), axis1=1, axis2=2)

        dif_x = self.observations.reshape(self.T,self.obs_dim,1) - np.matmul(self.C.reshape(1,self.obs_dim,self.hid_dim), self.smoothed_state_mean.reshape(self.T,self.hid_dim,1))
        E_x += np.matmul(dif_x.transpose(0,2,1), np.matmul(inv_cov_2.reshape(1,self.obs_dim,self.obs_dim), dif_x)).reshape(self.T)
        q_x = -(self.T/2)*np.log(np.linalg.det(self.cov_2))
        q_x += -(1/2)*np.sum(E_x, axis=0)

        loglikeli = q_0 + q_z + q_x

        return loglikeli

    def em(self, observations=None, itMax=10):
        self._initialize_params()
        if observations is not None:
            self.set_input(observations)
        for i in range(itMax):
            self.filter()
            self.smooth()
            self._expectation()
            self._maximization()

    def sieve(self):
        return