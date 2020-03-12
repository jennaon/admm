import argparse
import pdb
import numpy as np
from agent import Robot
import matplotlib.pyplot as plt
from run_admm import ADMMSolver
#
A =  np.array([[.1, .2],[.3, .4]])
B  = np.array([[1],[1]])

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

def simulate(robots, start):
    K = len(robots)
    T = robots[0].u.shape[0]
    dim = start.shape[0]
    x = np.zeros((dim*(T+1),K))
    for k in range(K):
        robot = robots[k]
        robot.control.append(robot.u[0])
        x[:2,[k]] = start[:,[k]]
        # pdb.set_trace()
        for t in range(1,T+1):
            x[dim*t:dim*(t+1),[k]] = robot.A @ x[dim*(t-1):dim*(t),[k]] + \
                            robot.B @ np.expand_dims(robot.u[t-1],axis=1)
    # pdb.set_trace()
    return x
def process_x(x):
    M,N = x.shape
    xx = np.zeros((int(M/2),N))
    xy = np.zeros_like(xx)
    for i in range(int(M/2)):
        xx[[i],:] = x[[2*i],:]
        xy[[i],:] = x[[2*i+1],:]
    # pdb.set_trace()
    return xx, xy

def make_plots(x,goals,iter):
    plt.figure()
    T,K = x.shape
    time = np.linspace(0,T-1,T)
    filename = './results/traj_iter'+str(iter)+'.png'
    x,y = process_x(x)
    for k in range(K):
        lbl ='robot'+str(k)
        # pdb.set_trace()
        plt.scatter(x[:,k],y[:,k],label=lbl,alpha=0.5)
    plt.legend()
    # plt.saveas(filename)

    # plt.show()


def simulate_and_plot(robots,inits, goals,iter):
    x=simulate(robots,inits)
    make_plots(x,goals,iter)


def main():
    parser = argparse.ArgumentParser(description='Lucky charm ADMM')
    parser.add_argument('--output', type=str, required=False, help='location to store results')
    parser.add_argument('--config', type=str, required=False, help='parameters')

    parser.add_argument('--max-iter', type=int, default=500,
                        help='max ADMM iterations (termination conditions, default: 500) ')
    parser.add_argument('--max-steps', type=int, default=50,
                        help='max steps that agents can take to reach the goal (default:50)')
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
    goals = np.array([[4,4],[2,0],[0,0],[1,1]]).T
    col,M_part = build_M(H=args.horizon, dim=args.dim,K=args.num_agents,inits=inits)

    np.set_printoptions(precision=2,suppress=True)

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
    paths = inits
    controls=None
    admm = ADMMSolver(robots=robots,
                     K = args.num_agents,
                      max_iter=args.max_iter)

    traj_count =0
    start = inits
    track_results =np.zeros((1,args.num_agents))
    np.set_printoptions(precision=2, suppress=True)
    while True:
        admm.solve()
        if np.mod(traj_count,10) ==0:
            print('iter %d .... ' %traj_count)
            pdb.set_trace()
        x=simulate(robots,start) #u is appended here
        next_pos = x[2:4,:]
        paths=np.vstack((paths,next_pos))

        #update starting positions for all robots
        for k in range(args.num_agents):
            robot = robots[k]
            # pdb.set_trace()
            robot.x0 = next_pos[:,[k]]

            track_results[0,k] = int(np.linalg.norm(robot.x0 - goals[:,[k]],2) < 0.1)

        if np.sum(track_results) >=2:
            print('success lol')
            break
        if traj_count >=args.max_steps:
            print('traj failed to converge, loop broken by the safety counter')
            break
        traj_count +=1

    pdb.set_trace()
    make_plots(paths,goals,iter)
    print('')



if __name__ == '__main__':
    main()
