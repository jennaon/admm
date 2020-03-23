import scipy as sp
import numpy as np
import pdb


from scipy.optimize import minimize

class Robot():
    def __init__(self,dim,inits,goals,A,B,u0,umax,rho,K,index,H,method):
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
        # self.u0=np.float64(np.random.randint(-2,5,size=(self.H*self.K,1)))
        self.u0 =u0
        # self.u0 = np.ones((self.H,1))
        self.control=[]
        self.u  = np.copy(self.u0)
        self.u_prev = np.zeros_like(self.u0)
        self.lambd= np.zeros_like(np.vstack((self.u0,self.u0)))
        self.rho = rho
        self.init_M()

        self.neighbors_dict={}
        self.W = np.zeros((self.dim,(self.H-1)*self.dim))
        self.W = np.hstack((self.W,np.eye(2))) #x' = Wx propagation matrix
        self.W =np.kron(np.eye(self.K,dtype=int),self.W)
        # pdb.set_trace()
        self.method = method
        self.distance_cost=[]
        self.reg_cost=[]
        self.umax =umax

    def init_M(self):
        M = np.zeros((self.H*self.dim,self.H*self.dim))
        for i in range(0,M.shape[0],self.dim):
            # val = int(i/2+1)-1
            M[i:i+self.dim,i:i+self.dim]=self.B
            # pdb.set_trace()

            for j in range(i-1,0,-self.dim):
                # print(j)
                M[i:i+self.dim,j-self.dim+1:j+1]=self.A @ M[i:i+self.dim,j+1:j+1+self.dim]

        col = np.zeros((self.dim*(self.H+1),self.dim))
        col[0:2,0:2]=np.eye(2)
        for i in range(self.H):
            col[self.dim*(i+1):self.dim*(i+2),:] = self.A @ col[self.dim*(i):self.dim*(i+1),:]
        # pdb.set_trace()
        # self.M = M
        self.M = np.kron(np.eye(self.K,dtype=int),M)
        # self.col = col[2:,:].reshape(-1,1) # propagation of how each robot propagates its init position thru time
        self.col = np.kron(np.eye(self.K),col[2:,:])

    def get_neighbors(self):
        # return np.mod([self.index-1+self.K,self.index+1],self.K)
        neigh = np.linspace(0,self.K-1,self.K,dtype=np.int32)
        return list(np.hstack((neigh[:self.index],neigh[self.index+1:])))

    def send_info(self):
        # return self.u
        return [self.u, self.lambd]

    def augmented_lagrangian(self,u):
        # u=u.reshape(-1,self.K)
        # u = np.expand_dims(u,axis=1)
        u=u.reshape(-1,1)
        # pdb.set_trace()
        reach_goal =self.W @(self.col @ self.inits + self.M @ u )-self.goals
        regularization = 0
        # print('final pos:')
        # print(self.W @(self.col @ self.inits + self.M @ u ))
        self.away_from_the_goal=( np.linalg.norm(reach_goal,2) **2)

        for j in range(self.K):
            if j == self.index:
                pass #myself, skip
            else:
                uj_prev = self.neighbors_dict[j][0]
                regularization += self.rho/2.0 * np.linalg.norm(u-(self.u_prev+uj_prev)/2 )**2
        # pdb.set_trace()
        self.regularization=regularization
        umax_const = np.abs(u) - (1/self.K) * np.ones_like(u) * self.umax

        # pdb.set_trace()
        # cost = np.linalg.norm(reach_goal,2) ** 2+ regularization + ((u.T @ self.lambd)[0,0]) #original master
        cost = np.linalg.norm(reach_goal,2) ** 2+ \
                self.rho / (4*len(self.neighbors_dict.keys())) * np.linalg.norm(
                            2.0/self.rho *regularization - \
                            1/self.rho * self.lambd[:len(self.u)] +\
                            1/self.rho * umax_const ,2)**2

        print('distacne cost:%.2f, regularize %.2f'%(0.5 * np.linalg.norm(reach_goal,2) ** 2,regularization))
        self.distance_cost.append(0.5 * np.linalg.norm(reach_goal,2) ** 2)
        self.reg_cost.append(regularization)
        # return (1/self.K)* cost
        return cost

    def primal_update(self,iter):
        # self.rho = self.rho/
        result = sp.optimize.minimize(self.augmented_lagrangian,
                                    x0=self.u0,
                                    method='Nelder-Mead',
                                    tol=0.1)#,
        self.cost=result['fun']
        # pdb.set_trace()
        self.u = np.expand_dims(result['x'],axis=1)
    #
    # def primal_update(self,iter):
    #     Ai = self.W @ self.M
    #     bi = self.goals-self.W @(self.col @ self.inits)
    #     # pdb.set_trace()
    #     avg = 0
    #     for j in self.neighbors_dict.keys():
    #          uj_prev = self.neighbors_dict[j]
    #          avg += self.u + uj_prev
    #     new_u=(1/self.K) * np.linalg.solve((1.0) * ((1/2.0) * (Ai.T @ Ai + self.rho * self.K * np.eye(Ai.shape[1]))),
    #                                             (self.rho/(iter+.01)) *avg + 2*Ai.T @bi - self.lambd)
        # print('terminal position: ')
        # print(Ai@new_u)
        # self.u = new_u


    def dual_update(self):
        # pdb.set_trace()
        new_lambd=self.lambd[:len(self.u)]
        for j in self.neighbors_dict.keys():
            new_lambd += self.rho * (self.lambd[len(self.u):] - self.neighbors_dict[j][1][len(self.u):])
            # new_lambd +=  self.rho*( np.vstack(self.u- self.neighbors_dict[j],
            #                                     self.u))
        new_nu = np.zeros_like(self.u)
        for j in self.neighbors_dict.keys():
            new_nu += self.lambd[len(self.u):] + self.neighbors_dict[j][1][len(self.u):]
        new_nu += 1/self.rho * (self.lambd[:len(self.u)]  +  np.abs(self.u) - (1/self.K)* np.ones_like(self.u) * self.umax)



        self.lambd = np.vstack((new_lambd,1/(2*len(self.neighbors_dict.keys())) * new_nu))
