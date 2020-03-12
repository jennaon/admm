import scipy as sp
import numpy as np
import pdb


from scipy.optimize import minimize

class Robot():
    def __init__(self,dim,goal,A,B,rho,K,index,H,M_part,col):
        self.A = np.array([[1, 1],[0, 1]])
        self.B = np.array([[1],[1]])
        self.dim=dim
        self.bdim = self.B.shape[0]
        # self.M = np.zeros((H*dim,(H)))
        self.goal =goal
        self.K = K
        self.index = index
        self.H = H
        self.safety = 2
        self.traj=[]
        # self.u0= np.vstack((np.ones((self.dim,1)),
        #                    np.zeros((self.H-1,1))))
        # self.u0 = np.ones((self.H,1))
        self.u0 = np.float64(np.random.randint(-2,5,size=(self.H,1)))

        self.u = np.zeros_like(self.u0)
        self.u_prev = np.zeros_like(self.u)
        self.lambd= np.zeros_like(self.u)
        self.rho = rho
        self.init_M(col, M_part)
        self.neighbors = None
        self.neighbors_dict={}
        self.eps = 1 #termination criteria
        self.W = np.zeros((self.dim,(self.H-1)*self.dim))
        self.W = np.hstack((self.W,np.eye(2))) #x' = Wx propagation matrix

    def init_M(self, col, M_part):
        # self.init = col[:2,[self.index]]
        self.inits = []
        for i in range(self.K):
            self.inits.append(col[:,self.dim*i:self.dim*(i+1)])
        self.col = col
        # pdb.set_trace()
        # self.M = np.hstack((col[self.dim:,self.dim*self.index:self.dim*(self.index+1)],
        #                     M_part[self.dim:,:]))
        self.M = M_part[2:,:]
        # print(' ')

    def get_neighbors(self):
        # return np.mod([self.index-1+self.K,self.index+1],self.K)
        neigh = np.linspace(0,self.K-1,self.K,dtype=np.int32)
        return list(np.hstack((neigh[:self.index],neigh[self.index+1:])))


    def send_info(self):
        return [self.u, self.M]

    def augmented_lagrangian(self,u, u_prev ):
        #gotta enforce initial conditions!!
        neighbors = self.get_neighbors()
        # cost =
        regularization= 0
        init_position = 0

        # pdb.set_trace()
        cost=0.5 * np.linalg.norm(self.W@self.M@u-self.goal,2 )**2
        for j in self.neighbors_dict.keys():
            collision_avoidance = 0
            uj, Mj = self.neighbors_dict[j]
            distance = np.expand_dims(self.M@u,axis=1) +self.col[:,[self.index]]\
                        - Mj@uj-self.col[:,[j]] # 2Tx1 matrix
            # pdb.set_trace()
            for t in range(self.H):
                # pdb.set_trace()
                dist = np.linalg.norm(distance[self.dim*t:self.dim*(t+1)],2)
                collision_avoidance += self.lambd[t,0]*(self.safety**2-dist)
            # print(collision_avoidance)
            # += np.linalg.norm(np.matmul(self.M-Mj,u),2)**2
            # collision_avoidance +=1.0/(np.linalg.norm(np.matmul(self.M-Mj,u),2)**2+.001)
            # pdb.set_trace()
            regularization += np.linalg.norm(u-(u_prev+uj)/2.0,2)**2
            cost +=collision_avoidance
        cost+= init_position+ self.rho*regularization
        # print(cost)
        # print('lambda : %.3f'%(self.lambd))
        return cost

    def primal_update(self,method='CG'):
        result = sp.optimize.minimize(self.augmented_lagrangian,
                                    x0=self.u0,
                                    args=(self.u_prev),
                                    method=method)#,
                                    # tol=0.001)
        # pdb.set_trace()
        self.cost=result['fun']
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
