import numpy as np
# import matplotlib.pyplot as plt
import time

import sys

import argparse

parser = argparse.ArgumentParser(description = "give random seed")
parser.add_argument("--seed", type = int, default = 0, help = 'random seed')
parser.add_argument("--T", type = int, default = 1000, help = 'random seed')
parser.add_argument("--d", type = int, default = 2, help = 'd')
parser.add_argument("--L", type = int, default = 2, help = 'L')
parser.add_argument("--sample", type = int, default = 0, help = 'whether sample or not')
args = parser.parse_args()

seed = args.seed
T = args.T
d = args.d
L = args.L
sample =args.sample

# mean of the parameters
A_mean = {}
for i in range(L-1):
    for x in range(0,d):
        A_mean[(i,x)] = np.ones((d+1,d+1))/(d+1)**2
    
AN_mean = np.ones((d+1,d+1))/(d+1)**2

N = L*d+1

# params = {'mathtext.default': 'regular' }  
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams.update(params)

if sample:
    A = {}
    for i in range(L-1):
        for x in range(0,d):
            A[(i,x)] = np.random.normal(loc=A_mean[(i,x)], scale=0.01)
        
    AN = np.random.normal(loc=AN_mean, scale=0.01)
else:
    A = A_mean
    AN = AN_mean


# generate values according to the intervention
def generate_X(At):
    """
    At: intervention
    output: vector X
    """
    # store values
    X=[[] for i in range(L+1)]
    for i in range(L):
        X[i] = np.zeros(d)
    X[L] = np.zeros(1)

    # layer 1
    for i in range(d):
        X[0][i] = np.random.normal(loc=1.0, scale=1.0, size=1)

    # except last layer
    for j in range(L-1):
        for i in range(d):
            X[j+1][i] = np.matmul(np.matmul(np.append(X[j],At[j][i]),A[(j,i)]),np.append(X[j],At[j][i])) + np.random.normal(loc=1.0, scale=1.0, size=1)
            # X[i]= sum(b[i-d]*X[:d])+epsilon[i]

    #reward node
    X[L] = np.matmul(np.matmul(np.append(X[L-1],At[L-1]),AN),np.append(X[L-1],At[L-1]))+ np.random.normal(loc=1.0, scale=1.0, size=1)
    return X

# for estimating mean
def generate_Xreward(A,AN,At,seed):
    """
    A: parameters except last layer
    AN: parameters for last layer
    At: intervention
    seed: random seeds
    output: reward
    """

    # Save the current state
    state = np.random.get_state()
    np.random.seed(seed)

    # store values
    X=[[] for i in range(L+1)]
    for i in range(L):
        X[i] = np.zeros(d)
    X[L] = np.zeros(1)

    # layer 1
    for i in range(d):
        X[0][i] = np.random.normal(loc=1.0, scale=1.0, size=1)

    # except last layer
    for j in range(L-1):
        for i in range(d):
            X[j+1][i] = np.matmul(np.matmul(np.append(X[j],At[j][i]),A[(j,i)]),np.append(X[j],At[j][i])) + np.random.normal(loc=1.0, scale=1.0, size=1)

    #reward node
    X[L] = np.matmul(np.matmul(np.append(X[L-1],At[L-1]),AN),np.append(X[L-1],At[L-1]))+ np.random.normal(loc=1.0, scale=1.0, size=1)

    # Restore the previous state
    np.random.set_state(state)
    return X[L]



def reconstruct_A(vec_A,d_A):
    '''
    construct A matrix from the vector d_A
    '''
    row_idx, col_idx = np.triu_indices(d_A,0)
    hat_A = np.zeros((d_A,d_A))
    hat_A[row_idx, col_idx] = vec_A
    hat_A[col_idx, row_idx] = vec_A
    return hat_A


def estimate_calculate_UCB(tilde_A,tilde_A_N,T=1000):
    """
    estimate UCBa(t)
    """
    UCB = np.full((2**((L-1)*d+1)), 0.)
    
    for action in range(2**((L-1)*d+1)):
        # get binary rep of action
        action_binary = [int(s) for s in "{0:b}".format(action)]
        action_binary[:0]=[0]*(((L-1)*d+1)-len(action_binary))
        
        AA = action_binary.copy()

        At=[[] for i in range(L)]
        for i in range(L-1):
            At[i] = [AA.pop(0) for _ in range(d)]
        At[L-1] = AA.pop(0)

        result = []
        for seed in range(T):
            result.append(generate_Xreward(tilde_A, tilde_A_N, At, seed))
        UCB[action] = np.mean(result)
    return UCB


UCB = estimate_calculate_UCB(A, AN,T=10000)
regret_action = -UCB + np.max(UCB)

def GCB(N,T):
    """
    N: number of nodes
    T: time horizon
    """
    
    #initial the estimators
    
    d_A = d+1
    d_vec = int(d_A*(d_A+1)/2)

    b_matrix = {}
    summ = {}
    M = {}
    #estimator of A
    hat_A = {}
    hat_A_vec = {}
    for j in range(L-1):
        for i in range(d):
            hat_A[(j,i)] = np.zeros((d_A,d_A))
            hat_A_vec[(j,i)] = np.zeros(d_vec,)
            b_matrix[(j,i)]  = np.zeros((d+1,d+1))
            summ[(j,i)] = np.zeros(((d+1)**2,(d+1)**2))
            M[(j,i)] = np.zeros((d_A,d_A))

    hat_A_N     = np.zeros((d_A,d_A)) 
    hat_A_N_vec = np.zeros(d_vec,)
    b_matrix_N  =  np.zeros((d+1,d+1)) 
    summ_N      = np.zeros(((d+1)**2,(d+1)**2)) 
    M_N         = np.zeros((d_A,d_A))

    # variables for saving the historical results
    regret = np.zeros(T)
    actions = np.zeros(T)

    row_idx, col_idx = np.triu_indices(d+1,0)

    for t in range(T):
        try:
            # sample weights according to posterior
            tilde_A = {}
            for j in range(L-1):
                for i in range(d):
                    tilde_A[(j,i)] = reconstruct_A(np.random.multivariate_normal(hat_A_vec[(j,i)],np.linalg.inv(summ[(j,i)][row_idx*(d+1)+col_idx,:][:,row_idx*(d+1)+col_idx])),d_A)

            tilde_A_N =reconstruct_A(np.random.multivariate_normal(hat_A_N_vec,np.linalg.inv(summ_N[row_idx*(d+1)+col_idx,:][:,row_idx*(d+1)+col_idx])),d_A)

            # store UCB for all actions
            UCB = estimate_calculate_UCB(tilde_A,tilde_A_N)
         
            # choose action that maxmize the UCB
            At = np.argmax(UCB)
        except:
            At = np.random.randint(0,2**((L-1)*d+1))
        At_binary = [int(s) for s in "{0:b}".format(At)]
        At_binary[:0]=[0]*((L-1)*d+1-len(At_binary))

        AA = At_binary.copy()
        #construct At
        At_list=[[] for i in range(L)]
        for i in range(L-1):
            At_list[i] = [AA.pop(0) for _ in range(d)]
        At_list[L-1] = AA.pop(0)
        

        # regret of action At
        regret[t] = regret_action[At]
        actions[t] = At
        # print(t,regret[t],actions[t])

        # generate possible samples
        X=generate_X(At_list)

        #update parameters by solving the lienar system
        for j in range(L-1):
            for i in range(d):
                tmp = np.append(X[j],At_list[j][i]).reshape(-1, 1)
                b_matrix[(j,i)] += X[j+1][i] * np.dot(tmp, tmp.T)
                b = b_matrix[(j,i)][row_idx, col_idx].flatten()
                matrix = np.dot(tmp, tmp.T)
                summ[(j,i)] += np.dot(matrix.flatten().reshape(-1, 1),matrix.flatten().reshape(-1, 1).T)
                result = summ[(j,i)] *2


                arr = np.arange(0, d+1) * (d+1)
                # Multiply non-diagonal elements of A by 2
                result[:,arr] = result[:,arr]/2

                M[(j,i)] = result[row_idx*(d+1)+col_idx,:][:,row_idx*(d+1)+col_idx]
                try:
                    solution = np.linalg.solve(M[(j,i)],b)
                    hat_A_vec[(j,i)] = solution
                    #re-construct the matrix
                    hat_A[(j,i)][row_idx, col_idx] = solution
                    hat_A[(j,i)][col_idx, row_idx] = solution
                except:
                    hat_A[(j,i)] = np.zeros((d_A,d_A))

        #update parameters for the final layer
        tmp = np.append(X[L-1],At_list[L-1]).reshape(-1, 1)
        b_matrix_N += X[L] * np.dot(tmp, tmp.T)
        b = b_matrix_N[row_idx, col_idx].flatten()
        matrix = np.dot(tmp, tmp.T)
        summ_N += np.dot(matrix.flatten().reshape(-1, 1),matrix.flatten().reshape(-1, 1).T)
        result = summ_N *2


        arr = np.arange(0, d+1) * (d+1)
        # Multiply non-diagonal elements of A by 2
        result[:,arr] = result[:,arr]/2

        M_N = result[row_idx*(d+1)+col_idx,:][:,row_idx*(d+1)+col_idx]
        try:
            solution = np.linalg.solve(M_N,b)
            hat_A_N_vec = solution
            #re-construct the matrix
            hat_A_N[row_idx, col_idx] = solution
            hat_A_N[col_idx, row_idx] = solution
        except:
            hat_A_N = np.zeros((d_A,d_A))

    return regret, actions


np.random.seed(2023)

#runing the trail
avgRegret_GCB=np.zeros(T)

iteration=1
for i in range(iteration):
    t1=time.time()
    print(i)
    regret_GCB, _ = GCB(N, T)
    avgRegret_GCB += regret_GCB / iteration
    t2=time.time()
    print(t2-t1)
    sys.stdout.flush()
    

# plt.rc('font',size=17)
# plt.rc('axes',titlesize=17)
# plt.rc('axes',labelsize=17)
# plt.rc('xtick',labelsize=16)
# plt.rc('ytick',labelsize=16)
# plt.rc('legend',fontsize=17)
# plt.rc('figure',titlesize=17)

# plt.figure()
# plt.plot(np.cumsum(avgRegret_GCB),linewidth=2)
# plt.grid()
# plt.ylabel('Cumulated Regret')
# plt.xlabel('Number of Iterations')
# plt.legend(["GCB"])
# plt.savefig('./result/plot/regret_qua_d%d_L%d.pdf'%(d,L),bbox_inches='tight',dpi=200)
# plt.close()
        
np.save('./result/data/regret_TS_qua_d%d_L%d'%(d,L), avgRegret_GCB)




