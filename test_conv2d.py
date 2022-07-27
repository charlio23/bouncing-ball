import time

import torch.nn
import torch.nn.functional as F

def get_sparse_kernel_matrix(K, h_X, w_X, s=1):

    # Assuming no channels and stride == 1.
    # Convert the kernel matrix to sparse matrix (dense matrix with lots of zeros in fact).
    # This is a little bit brain-twisting.
    """
        n_out, n_in, h_K, w_K = K.shape
        h_Y, w_Y = (h_X - h_K)//s + 1, (w_X - w_K)//s + 1
        W = torch.zeros((n_out,h_Y * w_Y, h_X * w_X * n_in))
        for i in range(h_Y):
            for j in range(w_Y):
                for ii in range(h_K):
                    for jj in range(w_K):
                        for kk in range(n_in):
                            W[:,i * w_Y + j, (s*i * w_X + s*j + ii * w_X + jj)*n_in + kk] = K[:,kk, ii, jj]
        return W.reshape(n_out*h_Y * w_Y, h_X * w_X * n_in)
    """
    n_in, n_out, h_K, w_K = K.shape
    h_Y, w_Y = (h_X - h_K)//s + 1, (w_X - w_K)//s + 1
    W = torch.zeros((h_Y * w_Y,n_in, h_X * w_X, n_out))
    for i in range(h_Y):
        for j in range(w_Y):
            for ii in range(h_K):
                for jj in range(w_K):
                        W[i * w_Y + j,:, s*i * w_X + s*j + ii * w_X + jj,:] = K[:,:, ii, jj]
    return W.reshape(h_Y * w_Y*n_in, h_X * w_X*n_out).T

def get_indices_sparse_kernel_matrix(K, h_X, w_X, s=1):

    # Assuming no channels and stride == 1.
    # Convert the kernel matrix to sparse matrix (dense matrix with lots of zeros in fact).
    # This is a little bit brain-twisting.

    _, _, h_K, w_K = K.shape
    h_Y, w_Y = (h_X - h_K)//s + 1, (w_X - w_K)//s + 1
    index_matrix = []
    index_kernel = []
    for i in range(h_Y):
        for j in range(w_Y):
            for ii in range(h_K):
                for jj in range(w_K):
                    index_matrix.append( (i * w_Y + j)*h_X*w_X +  s*i * w_X + s*j + ii * w_X + jj)
                    index_kernel.append(ii*w_K + jj)
    return index_matrix, index_kernel


K = torch.arange(32*1*6*6).reshape((32,1,6,6)).float()
X = torch.randn((32,14,14))

torch.set_printoptions(threshold=10_000)
a = F.conv_transpose2d(input=X.reshape((1,32,14,14)),weight=K.reshape((32,1,6,6)), stride=2, padding=0)
print(a.size())
M = get_sparse_kernel_matrix(K,32,32,s=2)
b1 = torch.matmul(M,X.permute(1,2,0).flatten()).reshape(1,32,32,1).permute(0,3,1,2)

print(a-b1)


start = time.time()
for i in range(100):
    a = F.conv_transpose2d(input=X.reshape((1,32,14,14)),weight=K.reshape((32,1,6,6)), stride=2, padding=0)
print('Torch Conv2D time:', time.time()-start)

start = time.time()
for i in range(100):
    M = get_sparse_kernel_matrix(K,32,32,s=2)
    b1 = torch.matmul(M,X.permute(1,2,0).flatten()).reshape(1,32,32,1).permute(0,3,1,2)
print('Matmul Conv2D time:', time.time()-start)
start = time.time()
index_matrix, index_kernel = get_indices_sparse_kernel_matrix(K,32,32,s=2)
n_in, n_out, h_K, w_K = K.shape
h_X = w_X = 32
h_Y, w_Y = (32 - h_K)//2 + 1, (32 - w_K)//2 + 1
for i in range(100):
    M = torch.zeros((n_in, n_out, h_Y * w_Y*h_X * w_X))
    M[:,:,index_matrix] = K.reshape(n_in,n_out,-1)[:,:,index_kernel]
    M = M.reshape((n_in, n_out, h_Y * w_Y, h_X * w_X)).permute(2,0,3,1).reshape(h_Y * w_Y*n_in, h_X * w_X*n_out).T
    b2 = torch.matmul(M,X.permute(1,2,0).flatten()).reshape(1,32,32,1).permute(0,3,1,2)
print('Matmul Conv2D time:', time.time()-start)

