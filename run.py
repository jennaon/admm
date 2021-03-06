import argparse
import pdb
import numpy as np
from agent import Robot
import matplotlib.pyplot as plt
from run_admm import ADMMSolver

np.random.seed(1234)
# A =  np.array([[.2, .5],[.3, .4]])
# B  = np.array([[.4],[.6]])
A =  np.array([[1, 0],[0, 1]])
B  = np.array([[1, 0],[0, 1]])

# def simulate_the_rest_and_plot(X,U,all_u,inits,goals,iter):
#     dim = 2
#     # pdb.set_trace()
#     K = int(inits.shape[0]/dim)
#     for i in range(1,all_u.shape[0]):
#         new_u=all_u[[i],:]
#         U=np.vstack((U,new_u))
#         new_pos=A@X[-1*dim:] + B@new_u
#         X = np.vstack((X,new_pos))
#         # pdb.set_trace()
#     make_plots(X,inits.reshape(-1,K,order='F'),
#                 goals.reshape(-1,K,order='F'),
#                 iter)


def process_x(x):
    M,N = x.shape
    xx = np.zeros((int(M/2),N))
    xy = np.zeros_like(xx)
    for i in range(int(M/2)):
        xx[[i],:] = x[[2*i],:]
        xy[[i],:] = x[[2*i+1],:]
    # pdb.set_trace()
    return xx, xy

def make_plots(X,inits,goals,iter,how_close,dir):
    fig=plt.figure()
    ax=plt.subplot(111)
    T,K = X.shape
    filename = dir+str(iter)+'.png'
    x,y = process_x(X)
    print(x)
    print(y)
    cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for k in range(K):
        lbl ='robot'+str(k)
        goal_lbl = lbl + ' goal'
        start_lbl = lbl + ' start'
        # pdb.set_trace()
        # color=cmap(k)

        # plt.show()
        ax.scatter(inits[0,k],inits[1,k],c=cmap[k],marker='o',label=start_lbl)
        ax.scatter(goals[0,k],goals[1,k],c=cmap[k],marker='*',label=goal_lbl)
        # pdb.set_trace()
        ax.plot(x[:,k],y[:,k],label=lbl,c=cmap[k],alpha=0.5)
    box=ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # plt.legend()
    filename = dir+'traj_'+str(iter)+'.png'
    title = 'Total distance away from the goals=' + str(np.round(how_close,2))
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def make_dist_reg_plot(dist,reg,iter,dir):
    #regularization & distance cost plot
    fig=plt.figure()
    # pdb.set_trace()
    ax=plt.subplot(111)

    # dist = dists[0]
    time = np.linspace(0, len(dist)-1, len(dist))
    cst_lbl = 'distance from the goal'
    ax.plot(time, dist, label=cst_lbl)

    # reg = regs[0]
    reg_lbl = 'regularization cost'
    ax.plot(time, reg, label=reg_lbl)
    # plt.show()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
    legend=ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=False, shadow=False, ncol=5)
    title = 'Iterative optimization of ADMM'
    filename = dir+'regdist_'+str(iter)+'.png'
    fig.suptitle(title)
    plt.savefig(filename)
    plt.close()

def make_u_plot(u,umax,iter,dir):
    fig=plt.figure()
    K = u.shape[1]
    ux,uy=process_x(u)
    # uxmax,uymax = process_x(umax)
    time = np.linspace(0, len(ux)-1, len(ux))

    cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
    width = 0.4
    # pdb.set_trace()
    ax=plt.subplot(211)
    ax.axhline(umax,linestyle='--',color='r')
    ax.axhline(-umax,linestyle='--',color='r')
    for k in range(K):
        lbl ='robot'+str(k) + ' x'
        # ax.plot(time,uxmax[0,k],'r-', label='maximum u')
        if k!=0:
            newtime=time-width/2 + width/k
        else:
            newtime=time-width/2
        ax.bar(newtime,ux[:,k],width=width,label=lbl,color=cmap[k],alpha=0.8)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.8])
    legend=ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=False, shadow=False, ncol=5)

    plt.title('Optimized X direction control input')

    ax=plt.subplot(212)
    ax.axhline(umax,linestyle='--',color='r')
    ax.axhline(-umax,linestyle='--',color='r')
    for k in range(K):
        lbl ='robot'+str(k) + ' y'
        # ax.plot(time,uymax[0,k],'r-', label='maximum u')
        if k!=0:
            newtime=time-width/2 + width/k
        else:
            newtime=time-width/2
        ax.bar(newtime,uy[:,k],width=width,label=lbl,color=cmap[k],alpha=0.8)

    plt.title('Optimized Y direction control input')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.8])
    legend=ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=False, shadow=False, ncol=5)
    filename = dir+'u_'+ str(iter)+'.png'
    # fig.suptitle(title)
    plt.savefig(filename)
    plt.close()



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
    parser.add_argument('--umax', type=float, default=10.0,
                        help='Maximum command (default: 10)')
    parser.add_argument('--solver', type=str, default='Nelder-Mead',
                        help='solver choice: choose \'CG\' or \'Powell\' to begin. default: CG')

    args = parser.parse_args()

    # inits = np.array([[0],[2],[3],[2],[3],[3],[2],[0]])
    # goals = np.array([[2],[4],[0],[-1],[4],[4],[3],[1]])
    inits = np.array([[0],[0],[4],[4]])
    # goals = np.array([[3],[3],[1],[1]])
    goals = np.array([[20],[20],[0],[0]])
    # col,M_part = build_M(H=args.horizon, dim=args.dim,K=args.num_agents)
    dir='./test/'

    np.set_printoptions(precision=3,suppress=True)
    random_u0=np.float64(np.random.randint(-2,5,size=(args.horizon*args.num_agents*args.dim,1)))
    robots= []
    for i in range(args.num_agents):
        robots.append(Robot(A = A,
                            B=B,
                            dim=args.dim,
                            inits=inits,
                            goals=goals,
                            u0=random_u0,
                            umax=args.umax,
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
    X = inits.reshape(-1,args.num_agents,order='F')
    while True:
        admm.solve(traj_count)
        print('iter %d .... ' %traj_count)
        new_u=robots[0].u.reshape(-1,args.num_agents,order='F')[[0],:]
        pos = (robots[0].M @ robots[0].u + (robots[0].col @ inits) ).reshape(-1,args.num_agents,order='F')
        # pdb.set_trace()

        last_pos = pos[-args.dim:]
        # U=np.vstack((U,new_u))
        # new_pos=A@X[-args.dim:] + B@new_u
        # X = np.vstack((X,new_pos))
        # simulate_the_rest_and_plot(X,U,robots[0].u.reshape(-1,args.num_agents,order='F'),
        #                             inits,goals,traj_count)
        # pdb.set_trace()
        #update u0 & x0 values
        # for k in range(args.num_agents):
        #     robots[k].u0=np.copy(robots[0].u)
        #     # pdb.set_trace()
        #     robots[k].inits=np.copy(new_pos.reshape(-1,1))


        how_close=np.linalg.norm(last_pos-goals.reshape(-1,args.num_agents,order='F'))
        print('how close: %.2f'%how_close)
        make_plots(np.vstack((inits.reshape(-1,args.num_agents,order='F'),pos)),
                    inits.reshape(-1,args.num_agents,order='F'),
                    goals.reshape(-1,args.num_agents,order='F'),
                    traj_count,how_close,dir)

        #collect reg & dist data:
        # dists = []
        # regs=[]
        # for k in range(args.num_agents):
        #     dists.append(robots[k].distance_cost)
        #     regs.append(robots[k].reg_cost)
        make_dist_reg_plot(robots[0].distance_cost,
                            robots[0].reg_cost,
                            traj_count,dir)

        make_u_plot(robots[0].u.reshape(-1,args.num_agents,order='F'),
                    args.umax,
                    traj_count,dir)
        if how_close<.5:
            print('reached the goal')
            break

        if traj_count >=args.max_steps:
            print('traj failed to converge, loop broken by the safety counter')
            break
        traj_count +=1

    iter ='FINAL'

    print('')



if __name__ == '__main__':
    main()
