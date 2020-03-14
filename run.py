import argparse
import pdb
import numpy as np
from agent import Robot
import matplotlib.pyplot as plt
from run_admm import ADMMSolver
#
A =  np.array([[.2, .5],[.3, .4]])
B  = np.array([[1],[1]])

# def simulate(robots, xprev,u):
#
#     # K = len(robots)
#     # T = robots[0].u.shape[0]
#     # dim = start.shape[0]
#     # x = np.zeros((dim*(T+1),K))
#
#     #since ADMM converged, assumption is that the solution U is the same!
#     pdb.set_trace()
#     x = A@xprev
#
#     # for k in range(K):
#     #     robot = robots[k]
#     #     robot.control.append(robot.u[0])
#     #     x[:2,[k]] = start[:,[k]]
#     #     for t in range(1,T+1):
#     #         x[dim*t:dim*(t+1),[k]] = robot.A @ x[dim*(t-1):dim*(t),[k]] + \
#     #                         robot.B @ np.expand_dims(robot.u[t-1],axis=1)
#     # pdb.set_trace()
#     return A@xprev + B@u

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
        plt.plot(x[:,k],y[:,k],label=lbl,alpha=0.5)
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
    parser.add_argument('--rho', type=float, default=.01,
                        help='step size (default: .01)')
    parser.add_argument('--horizon', type=int, default=4, metavar='H',
                        help='TrajOpt horizon (default: 4)')
    parser.add_argument('--dim', type=int, default=2,
                        help='state space dimension (default: 2)')
    parser.add_argument('--solver', type=str, default='CG',
                        help='solver choice: choose \'CG\' or \'Powell\' to begin. default: CG')

    args = parser.parse_args()
    # with open(args.config, 'r') as f:
    #     config = eval(f.read())


    # inits = np.array([[1,0],[0,2],[4,4],[2,0]]).T
    # goals = np.array([[4,4],[2,0],[0,0],[1,1]]).T
    inits = np.array([[1,0],[0,2]]).T
    goals = np.array([[4,4],[2,0]]).T
    # col,M_part = build_M(H=args.horizon, dim=args.dim,K=args.num_agents)

    np.set_printoptions(precision=3,suppress=True)
    random_u0=np.float64(np.random.randint(-2,5,size=(args.horizon,args.num_agents)))
    robots= []
    for i in range(args.num_agents):
        robots.append(Robot(A = A,
                            B=B,
                            dim=args.dim,
                            inits=inits,
                            goals=goals,
                            u0=random_u0,
                            rho = args.rho,
                            K =args.num_agents,
                            index=i,
                            H = args.horizon,
                            method=args.solver)) # i should move this to run_admm

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
    U = np.ones((1,args.num_agents))
    X = inits
    while True:
        admm.solve()
        if np.mod(traj_count,10) ==0 and traj_count>0:
            print('iter %d .... ' %traj_count)
            pdb.set_trace()
        new_u=robots[0].u[[0],:]
        U=np.vstack((U,new_u))
        # A@xprev + B@u
        new_pos=A@X[-args.dim:] + B@new_u
        # simulate(robots,,new_u)
        X = np.vstack((X,new_pos))
        #update u0 values
        for k in range(args.num_agents):
            robots[k].u0=np.copy(robots[0].u)
            robots[k].inits=np.copy(new_pos)
        # paths=np.vstack((paths,next_pos))
        # pdb.set_trace()
        how_close=np.linalg.norm(new_pos-goals)
        print(how_close)

        if np.linalg.norm(new_pos-goals)<1:
            print('reached the goal')
            break

        #update starting positions for all robots
        # for k in range(args.num_agents):
        #     robot = robots[k]
        #     # pdb.set_trace()
        #     robot.x0 = next_pos[:,[k]]
        #
        #     track_results[0,k] = int(np.linalg.norm(robot.x0 - goals[:,[k]],2) < 0.1)
        #
        # if np.sum(track_results) >=2:
        #     print('success lol')
        #     break
        if traj_count >=args.max_steps:
            print('traj failed to converge, loop broken by the safety counter')
            break
        traj_count +=1

    pdb.set_trace()
    make_plots(paths,goals,iter)
    print('')



if __name__ == '__main__':
    main()
