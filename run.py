import argparse
import pdb
import numpy as np
from agent import Robot

A = np.array([[.1, .2],[.3, .4]])
B  = np.array([[0],[1]])

def build_M(H,dim,K,inits):
    M = np.zeros((H*dim,(H)))
    # M[0:2,0:2]=A
    # M[0:2,[2]]=B
    for i in range(0,M.shape[0],dim):
        val = int(i/2+1)-1
        # print('row %d, val=%d'%(i,val))
        M[i:i+dim,[val]]=B
        for j in range(val,0,-1):
            # print(j)
            M[i:i+dim,[j-1]]=np.matmul(A,M[i:i+dim,[j]])
    col = inits
    prev = inits
    this=np.zeros((dim,K))
    for i in range(H-1):
        for k in range(K):
            this[:,[k]] = np.matmul(A,prev[:,[k]])
        # pdb.set_trace()
        col = np.vstack((col, this))
        prev = this

    M = np.vstack((np.zeros((dim,M.shape[1])),M))
    return col,M

def simulate(robots, inits):
    K = len(robots)
    T = robots[0].u.shape[0]-1
    pdb.set_trace()
    x = np.zeros((T,K))
    for k in range(K):
        for t in range(T):
            pass

def plot():
    pass



def simulate_and_plot(robots,inits, goals,iter):
    simulate(robots,inits)
    plot()


def main():
    parser = argparse.ArgumentParser(description='Lucky charm ADMM')
    parser.add_argument('--output', type=str, required=False, help='location to store results')
    parser.add_argument('--config', type=str, required=False, help='parameters')

    parser.add_argument('--max-iter', type=int, default=500,
                        help='max ADMM iterations (termination conditions, default: 500) ')
    parser.add_argument('--num-agents', type=int, default=4, metavar='N',
                        help='number of robots (default: 4)')
    parser.add_argument('--rho', type=float, default=0.4,
                        help='step size (default: .4)')
    parser.add_argument('--horizon', type=int, default=4, metavar='H',
                        help='TrajOpt horizon (default: 4)')
    parser.add_argument('--dim', type=int, default=2,
                        help='state space dimension (default: 2)')

    args = parser.parse_args()
    # with open(args.config, 'r') as f:
    #     config = eval(f.read())


    inits = np.array([[1,0],[0,2],[4,4],[2,0]]).T
    goals = np.array([[4,4],[2,0],[0,0],[3,3]]).T
    col,M_part = build_M(H=args.horizon, dim=args.dim,K=args.num_agents,inits=inits)



    robots= []
    for i in range(args.num_agents):
        robots.append(Robot(A = A,
                            B=B,
                            dim=args.dim,
                            goal=goals[:,[i]],
                            rho = args.rho,
                            K =args.num_agents,
                            index=i,
                            H = args.horizon,
                            M_part = M_part,
                            col=col))

    rho_candidates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.8]
    solvers = ['Powell', 'CG']
    x = []

    count = 0
    result = np.zeros((args.num_agents,1))
    # pdb.set_trace()
    while True:
        if np.mod(count,10) ==0:
            print('iter %d .... ' %count)
            pdb.set_trace()
        for k in range(args.num_agents):
            #update neighbors
            print(k)
            # pdb.set_trace()
            neighbors=robots[k].get_neighbors()
            for j in neighbors:
                robots[k].neighbors_dict[j] =(robots[j].send_info())
        # pdb.set_trace()
        # simulate_and_plot(robots,inits, goals,count)

        for k in range(args.num_agents):
            robots[k].primal_update()
            robots[k].dual_update()
            # robots[k].step_forward()
        # pdb.set_trace()
        count +=1
        if count > args.max_iter:
            print('failed to converge, loop broken by the safety counter')
            break

        result = np.zeros((args.num_agents,1))
        if count > 10:
            for k in range(args.num_agents):
                result[k] = int(robots[k].compare_vals())
            print(np.sum(result))
            if np.sum(result) ==args.num_agents: #slack
                print('converged!!')
                break

    #simulate()
    for k in range(args.num_agents):
        pass
    pdb.set_trace()
    print('')



if __name__ == '__main__':
    main()
