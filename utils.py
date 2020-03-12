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

# M = np.hstack((col[:,[0]],M))#not 0 but for each robot
# M1 = np.hstack((col[:,[1]],M))#not 0 but for each robot

#         # print(M)
# pdb.set_trace()
# M = np.repeat(M,K,axis=2)

def get_neighbors(i):
    return np.mod([np.abs(i-1),i+1],K)#,
    #         np.hstack((col[:,[i]],M)),
    #         np.hstack((col[:,[np.mod(np.abs(i-1),K)]]],M)),
    #         np.hstack((col[:,[np.mod(np.abs(i+1),K)]]],M)))

u0 = np.zeros((H-1,1)) #for each robot
u0 = np.vstack((np.ones((dim,1)),u0))
u=u0
# lamdba = np.zeos((1,1))
lambd= 0.1
A = np.zeros((dim,(H)*dim))
A = np.hstack((A,np.eye(2)))
x0 =np.expand_dims( np.array([1.3, 0.7, 0.8, 1.9, 1.2]),axis=1)
u_prev = np.expand_dims(np.array([1.1, 0.5, 0.4, 1.1, .7]),axis=1)
rho = 0.1

def getM(i):
    return np.hstack((col[:,[i]],M))

def augmented_lagrangian(u, u_prev,lambd=.1,rho=.1, ):
    #gotta enforce initial conditions!!
    neighbors = get_neighbors(i)
    Mi =getM(i)
    cost = 0.5 * np.linalg.norm(A@Mi@u,2)**2
    collision_avoidance = 0
    regularization= 0
    for j in neighbors:
        Mj = getM(j)
        collision_avoidance += np.linalg.norm(np.matmul(Mi-Mj,u),2)**2
        # pdb.set_trace()
        regularization += np.linalg.norm(u-(u_prev+x0)/2,2)**2
    cost += collision_avoidance + regularization

    return cost

c = augmented_lagrangian(u0,x0)
pdb.set_trace()
result = sp.optimize.minimize(augmented_lagrangian,x0=u0,args=(x0,.5,.2),method='Powell')
print(' ')
# scipy.optimize.minimize()
