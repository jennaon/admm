import scipy as sp
import numpy as np
import pdb


from scipy.optimize import minimize

class Robot():
    def __init__(self,dim,goal,A,B,rho,K,index,H,M_part,col):
        self.A = np.array([[.1, .2],[.3, .4]])
        self.B = np.array([[0],[1]])
        self.dim=dim
        self.bdim = self.B.shape[0]
        # self.M = np.zeros((H*dim,(H)))
        self.goal =goal
        self.K = K
        self.index = index
        self.H = H
        self.safety = 0
        self.u0= np.vstack((np.ones((self.dim,1)),
                           np.zeros((self.H-1,1))))
        self.u = np.zeros_like(self.u0)
        self.u_prev = np.zeros_like(self.u)
        self.lambd= np.zeros_like(self.u)
        self.rho = rho
        self.init_M(col, M_part)
        self.neighbors = None
        self.neighbors_dict={}
        self.eps = 1 #termination criteria
        self.W = np.zeros((self.dim,(self.H)*self.dim))
        self.W = np.hstack((self.W,np.eye(2))) #x' = Wx propagation matrix

    def init_M(self, col, M_part):
        # pdb.set_trace()
        self.init = col[:2,[self.index]]
        self.M = np.hstack((col[:,[self.index]],M_part))

    def get_neighbors(self):
        return np.mod([self.index-1+self.K,self.index+1],self.K)

    def send_info(self):
        return [self.u, self.M]

    def augmented_lagrangian(self,u, u_prev ):
        #gotta enforce initial conditions!!
        neighbors = self.get_neighbors()
        # pdb.set_trace()
        # cost =
        regularization= 0
        collision_avoidance = 0
        init_position = 0
        for j in self.neighbors_dict.keys():
            # Mj = getM(j)
            uj, Mj = self.neighbors_dict[j]
            # pdb.set_trace()
            distance = self.M@u - Mj@u # 2Tx1 matrix
            for t in range(self.u.shape[1]):
                # if t<2:
                #     # pdb.set_trace()
                #     # init_position +=self.lambd[t,0]*np.linalg.norm( (np.expand_dims((self.M@u),axis=1)[:2]-self.init),2)
                #     init_position +=self.lambd[t,0]* np.abs(u[t]-1)**2
                # else:
                collision_avoidance += self.lambd[t,0]*(self.safety**2-np.abs(distance[t]))
            # print(collision_avoidance)
            # += np.linalg.norm(np.matmul(self.M-Mj,u),2)**2
            # collision_avoidance +=1.0/(np.linalg.norm(np.matmul(self.M-Mj,u),2)**2+.001)
            # pdb.set_trace()
            regularization += np.linalg.norm(u-(self.u_prev+uj)/2,2)**2
        cost  = 0.5 * np.linalg.norm(self.W@self.M@u-self.goal,2 )**2 + \
                        collision_avoidance + init_position+ \
                        self.rho*regularization
        # print(cost)
        # print('lambda : %.3f'%(self.lambd))
        return cost

    def primal_update(self,method='CG'):
        result = sp.optimize.minimize(self.augmented_lagrangian,
                                    x0=self.u0,
                                    args=(self.u),
                                    method=method)#,
                                    # tol=0.001)
        # pdb.set_trace()
        self.u_prev = self.u
        self.u = np.expand_dims(result['x'],axis=1)

    def dual_update(self):
        new_lambd=self.lambd
        # if self.neighbors is None:
        #     self.get_neighbors()
        for j in self.neighbors_dict.keys():
            # pdb.set_trace()
            new_lambd +=  self.rho*( self.u - self.neighbors_dict[j][0])
        self.lambd_prev = self.lambd
        self.lambd = new_lambd

    def compare_vals(self):
        # if self.neighbors is None:
        #     self.get_neighbors()
        deviation = 0
        for j in self.neighbors_dict.keys():
            deviation += np.linalg.norm(self.u - self.neighbors_dict[j][0])
        if deviation< self.eps: #uj
            return True
                # print('%d and %d converged'%(i,j))
        else :
            return False

    # def step_forwad(self):
    #     self.x_new = self.A@self.x + self.B @ self.u[-2:,1]
