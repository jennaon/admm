import scipy as sp
import numpy as np
import pdb


from scipy.optimize import minimize

class Robot():
    def __init__(self,dim,rho,K,index,H,M_part,col):
        self.A = np.array([[.1, .2],[.3, .4]])
        self.B = np.array([[0],[1]])
        self.dim=dim
        self.bdim = self.B.shape[0]
        # self.M = np.zeros((H*dim,(H)))
        self.lambd= 0
        self.K = K
        self.index = index
        self.H = H
        self.u0= np.vstack((np.ones((self.dim,1)),
                           np.zeros((self.H-1,1))))
        self.u = np.zeros_like(self.u)
        self.u_prev = np.zeros_like(self.u)
        self.rho = rho
        self.init_M(inits,col, M_part)
        self.neighbors = None
        self.eps = 2 #termination criteria

    def init_M(self, inits, col, M_part):
        self.M = np.hstack((col[:,[self.index]],M_part))

    def get_neighbors(self):
        print('fill out get_neighbors')
        self.neighors = {1:uj,Mj}
        return

    def send_info(self):
        return [self.u, self.M]

    def augmented_lagrangian(self,u, u_prev,lambd=.1,rho=.1, ):
        #gotta enforce initial conditions!!
        neighbors = get_neighbors(i)
        # Mi =getM(i)
        cost = 0.5 * np.linalg.norm(A@self.M@u,2)**2
        collision_avoidance = 0
        regularization= 0
        if self.neighbors is None:
            self.get_neighbors()
        for j in self.neighbors.keys():
            Mj = getM(j)
            collision_avoidance += np.linalg.norm(np.matmul(Mi-Mj,u),2)**2
            # pdb.set_trace()
            regularization += np.linalg.norm(u-(u_prev+x0)/2,2)**2
        cost += collision_avoidance + regularization

        return cost

    def primal_update(self,method='Powell'):
        result = sp.optimize.minimize(self.augmented_lagrangian,
                                    x0=self.u0,
                                    args=(self.u,.5,.2),
                                    method=method)
        self.u_prev = self.u
        self.u = result['x']

    def dual_update(self):
        new_lambd=self.lambd
        if self.neighbors is None:
            self.get_neighbors()
        for j in self.neighbors.keys():
            new_lambd += rho*( self.u - self.neighbors[j][0])
        self.lambd_prev = self.lambd
        self.lambd = new_lambd

    def compare_vals(self):
        if self.neighbors is None:
            self.get_neighbors()
        for j in self.neighbors.keys():
            if np.linalg.norm(self.u - self.neighbors[j][0]) < self.eps: #uj
                print('%d and %d converged'%(i,j))
        return 0
