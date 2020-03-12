import scipy as sp
import numpy as np
import pdb


from scipy.optimize import minimize

class Robot():
    def __init__(self,dim,A,B,rho,K,index,H,M_part,col):
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
        self.u = np.zeros_like(self.u0)
        self.u_prev = np.zeros_like(self.u)
        self.rho = rho
        self.init_M(col, M_part)
        self.neighbors = None
        self.neighbors_dict={}
        self.eps = 2 #termination criteria
        self.W = np.zeros((self.dim,(self.H)*self.dim))
        self.W = np.hstack((self.W,np.eye(2))) #x' = Wx propagation matrix

    def init_M(self, col, M_part):
        self.M = np.hstack((col[:,[self.index]],M_part))

    def get_neighbors(self):
        return np.mod([self.index-1+self.K,self.index+1],self.K)

    def send_info(self):
        return [self.u, self.M]

    def augmented_lagrangian(self,u, u_prev,lambd=.1,rho=.1, ):
        #gotta enforce initial conditions!!
        neighbors = self.get_neighbors()
        # pdb.set_trace()
        cost = 0.5 * np.linalg.norm(self.W@self.M@u,2)**2
        collision_avoidance = 0
        regularization= 0
        if self.neighbors_dict is None:
            self.get_neighbors()
        for j in self.neighbors_dict.keys():
            # Mj = getM(j)
            uj, Mj = self.neighbors_dict[j]
            # pdb.set_trace()
            collision_avoidance += np.linalg.norm(np.matmul(self.M-Mj,u),2)**2
            # pdb.set_trace()
            regularization += np.linalg.norm(u-(self.u_prev+uj)/2,2)**2
        cost += collision_avoidance + self.rho*regularization

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
        # if self.neighbors is None:
        #     self.get_neighbors()
        for j in self.neighbors_dict.keys():
            new_lambd += self.rho*( self.u - self.neighbors_dict[j][0])
        self.lambd_prev = self.lambd
        self.lambd = new_lambd

    def compare_vals(self):
        # if self.neighbors is None:
        #     self.get_neighbors()
        for j in self.neighbors_dict.keys():
            if np.linalg.norm(self.u - self.neighbors_dict[j][0]) < self.eps: #uj
                print('%d and %d converged'%(i,j))
        return 0
