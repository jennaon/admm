import scipy as sp
import numpy as np
import pdb


from scipy.optimize import minimize

class Robot():
    def __init__(self,dim,inits,goals,A,B,u0,rho,K,index,H,method):
        self.A = A
        self.B = B
        self.dim=dim
        self.bdim = self.B.shape[0]
        self.goals =goals
        self.inits = inits
        self.K = K
        self.index = index
        self.H = H
        self.safety =.5
        self.traj=[]
        # self.u0= np.vstack((np.ones((self.dim,K)),
        #                     np.float64(np.random.randint(-2,5,
        #                                         size=(self.H,K)))))
        self.u0=u0
        # self.u0 =
        # self.u0 = np.ones((self.H,1))
        self.control=[]
        self.u  = np.copy(self.u0)
        self.u_prev = np.copy(self.u0)
        self.lambd= np.zeros_like(self.u0)
        self.rho = rho
        self.init_M()

        self.neighbors_dict={}
        self.W = np.zeros((self.dim,(self.H-1)*self.dim))
        self.W = np.hstack((self.W,np.eye(2))) #x' = Wx propagation matrix
        self.W =np.kron(np.eye(self.K,dtype=int),self.W)
        # pdb.set_trace()
        self.method = method

    def init_M(self):
        M = np.zeros((self.H*self.dim,(self.H)))
        for i in range(0,M.shape[0],self.dim):
            val = int(i/2+1)-1
            M[i:i+self.dim,[val]]=self.B
            for j in range(val,0,-1):
                M[i:i+self.dim,[j-1]]=self.A @ M[i:i+self.dim,[j]]

        col = np.zeros((self.dim*(self.H+1),self.dim))
        col[0:2,0:2]=np.eye(2)
        for i in range(self.H):
            col[self.dim*(i+1):self.dim*(i+2),:] = self.A @ col[self.dim*(i):self.dim*(i+1),:]
        # self.M = M
        self.M = np.kron(np.eye(self.K,dtype=int),M)
        # pdb.set_trace()
        # self.col = col[2:,:].reshape(-1,1) # propagation of how each robot propagates its init position thru time
        self.col = np.kron(np.eye(self.K),col[2:,:])

    def get_neighbors(self):
        # return np.mod([self.index-1+self.K,self.index+1],self.K)
        neigh = np.linspace(0,self.K-1,self.K,dtype=np.int32)
        return list(np.hstack((neigh[:self.index],neigh[self.index+1:])))

    def send_info(self):
        return self.u

    def augmented_lagrangian(self,u):
        # u=u.reshape(-1,self.K)
        # u = np.expand_dims(u,axis=1)
        u=u.reshape(-1,1)
        # pdb.set_trace()
        reach_goal =self.W @(self.col @ self.inits + self.M @ u )-self.goals
        regularization = 0
        cost = 0.5 * np.linalg.norm(reach_goal,2) ** 2
        self.away_from_the_goal=(cost*2)


        for j in range(self.K):
            if j == self.index:
                pass #myself, skip
            else:
                uj_prev = self.neighbors_dict[j]
                regularization += np.linalg.norm(u-(self.u_prev+uj_prev)/2 )**2
        # pdb.set_trace()
        self.regularization=regularization
        cost += self.rho/2.0 * regularization +(u.T @ self.lambd)[0,0]

        # pdb.set_trace()
        '''
        todo: add collision avoidance
        first two values of u should not change over iteration
        '''
        return cost

    def primal_update(self):
        result = sp.optimize.minimize(self.augmented_lagrangian,
                                    x0=self.u0,
                                    method=self.method)#,
        self.cost=result['fun']
        # print('cost: %.3f'%self.cost)
        # pdb.set_trace()
        # return result['x'].reshape(-1,self.K)
        return np.expand_dims(result['x'],axis=1)

    def dual_update(self, new_u):
        # pdb.set_trace()
        new_lambd=self.lambd
        for j in self.neighbors_dict.keys():
            new_lambd +=  self.rho*( new_u- self.neighbors_dict[j])

        self.lambd_prev = self.lambd
        self.lambd = new_lambd

        self.u_prev = self.u
        self.u = new_u
