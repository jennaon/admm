import scipy as sp
import numpy as np
import pdb


from scipy.optimize import minimize, rosen

x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
# res = minimize(rosen, x0, method='Nelder-Mead', tol=1e-6)
# pdb.set_trace()
# print('')


A = np.array([[.1, .2],[.3, .4]])
B = np.array([[0],[1]])
H = 4 #horizon
dim = 2
bdim=B.shape[1]
K = 4 #number of robots
np.set_printoptions(precision=2,suppress=True)
inits = np.array([[1,0],[0,2],[4,4],[2,0]]).T
#build M
M = np.zeros((H*dim,(H)))
# M[0:2,0:2]=A
# M[0:2,[2]]=B
for i in range(0,M.shape[0],dim):
    val = int(i/2+1)-1
    print('row %d, val=%d'%(i,val))
    M[i:i+dim,[val]]=B
    for j in range(val,0,-1):
        print(j)
        M[i:i+dim,[j-1]]=np.matmul(A,M[i:i+dim,[j]])
col = inits
prev = inits
this=np.zeros((dim,K))
for i in range(H):
    for k in range(K):
        this[:,[k]] = np.matmul(A,prev[:,[k]])
    # pdb.set_trace()
    col = np.vstack((col, this))
    prev = this

M = np.vstack((np.zeros((dim,M.shape[1])),M))

M = np.hstack((col[:,[0]],M))#not 0 but for each robot

# for i in range(0,M.shape[0],dim):
#     print('row %d'%i)
#     M[i:i+dim, 0:2] = A**(i+1)
#     val = int((i+dim)/2)
#     print(val)
#     for j in range(2,val+2,bdim):
#         # pdb.set_trace()
#         # print(j)
#         M[i:i+dim,j:j+bdim] = np.matmul((A**(val-(j-2)-1)),  B)
#         print('%d th column, A**%d'%(j,(val-(j-2)-1)))
#         # print(M)
pdb.set_trace()
M = np.repeat(M,K,axis=2)
for k in range(K):
    M[0,:dim,k] =inits[:,k]
