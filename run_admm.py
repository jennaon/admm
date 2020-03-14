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

    def solve(self):
        count = 0
        result = np.zeros((self.K,1))
        while True:
            # if np.mod(count,2) ==0:
                # print('iter %d .... ' %count)
                # pdb.set_trace()
            for k in range(self.K):
                robot = self.robots[k]
                #update neighbors
                neighbors=robot.get_neighbors()
                for j in neighbors:
                    robot.neighbors_dict[j] =(self.robots[j].send_info())
            for k in range(self.K):
                robot = self.robots[k]
                robot.primal_update()
                robot.dual_update()
            count +=1
            if count > self.max_iter:
                print('admm failed to converge, loop broken by the safety counter')
                break

            result = np.zeros((self.K,1))
            if count > 10:
                for k in range(self.K):
                    result[k] = int(self.robots[k].compare_vals())
                if np.sum(result) ==self.K:
                    print('converged!!')
                    break
