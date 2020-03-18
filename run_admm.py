import argparse
import pdb
import numpy as np
from agent import Robot
import matplotlib.pyplot as plt

class ADMMSolver():
    def __init__(self,robots, K, max_iter):
        self.robots = robots
        self.K = K
        self.max_iter = max_iter
        self.threshold = .1

    def iterate_step(self,iter):
        # new_u=np.zeros((self.robots[0].u.shape[0],self.K))
        for k in range(self.K):
            robot = self.robots[k]
            robot.primal_update(iter)
            # print(new_u)
        for k in range(self.K):
            robot = self.robots[k]
            robot.dual_update()

    def update_my_data(self):
        for k in range(self.K):
            robot = self.robots[k]
            #update neighbors
            neighbors=robot.get_neighbors()
            for j in neighbors:
                uj =self.robots[j].send_info()
                robot.neighbors_dict[j] =uj

    def solve(self,iter):
        count = 0
        result = np.zeros((self.K,1))
        # print(np.hstack((self.robots[0].u,self.robots[1].u)))
        while True:
            print('admm iter %d .... ' %count)
            self.update_my_data()

            '''
            -update self.u0 as well for x0 in minimizer?
            -check whether this step makes sense for the best first update?
            -adjust rho over time
            -update u0 over time!!!!
            -include something that if admm fails to converge,
                try again with a different init etc
            '''
            self.iterate_step(iter)
            # pdb.set_trace()
            count +=1
            if count > self.max_iter:
                print('admm failed to converge, loop broken by the safety counter')
                break

            # print(np.hstack((self.robots[0].u,self.robots[1].u)))

            #check whether i converged

            compare=self.robots[0].u.reshape(-1,1)
            dev = 0
            for k in range(1,self.K):
                dev += np.linalg.norm(compare-self.robots[k].u.reshape(-1,1),2)
            if dev <self.threshold:
                # pdb.set_trace()
                print('ADMM converged :), iter: %d. below is the review:'%count)
                # for k in range(self.K):
                #     print('robot %d away from the goal by %.2f'%(k,self.robots[k].away_from_the_goal))
                break
