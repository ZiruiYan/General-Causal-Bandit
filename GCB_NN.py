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

# np.random.seed(seed)


def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

import torch
import torch.nn as nn
import torch.optim as optim

# Define the class for single layer NN
class NN(torch.nn.Module):    
    # Constructor
    def __init__(self):
        super(NN, self).__init__()
        self.linear_one = torch.nn.Linear(d+1, 1, bias = False)
        self.relu = nn.LeakyReLU(0.1)
    # prediction function
    def forward(self, x):
        layer_in = self.linear_one(x)
        return layer_in


# mean of the parameters
b_mean = {}
for i in range(1,2*L-1):
    if i%2==1:
        for j in range(d):
            for k in range(0, d+1):
                b_mean[(i,j,k)] = [0.5/np.sqrt(d+1)]*(d+1)
    else:
        for j in range(0,d):
            b_mean[(i,j)] = [0.5/np.sqrt(d+1)]*(d+1)

# final layer
i=2*L-1
j=0
for k in range(0, d+1):
    b_mean[(i,j,k)] = [0.5/np.sqrt(d+1)]*(d+1)

bN_mean = [0.5/np.sqrt(d+1)]*(d+1)

if sample:
    b= {}
    for i in range(1,2*L-1):
        if i%2==1:
            for j in range(d):
                for k in range(0, d+1):
                    b[(i,j,k)] = np.random.normal(loc=b_mean[(i,j,k)], scale=0.01)
        else:
            for j in range(0,d):
                b[(i,j)] = np.random.normal(loc=b_mean[(i,j)], scale=0.01)
    # final layer
    i=2*L-1
    j=0
    for k in range(0, d+1):
        b[(i,j,k)] = np.random.normal(loc=b_mean[(i,j,k)], scale=0.01)
    bN = np.random.normal(loc=bN_mean, scale=0.01)
else:
    b = b_mean 
    bN = bN_mean

#number of nodes
N = L*(2*d+1)+1



# params = {'mathtext.default': 'regular' }  
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams.update(params)




# generate values according to the intervention
def generate_X(At_list):
    """
    At_list: intervention
    output: vector X
    """
    # store values
    X=[[] for i in range(2*L+1)]
    for i in range(L):
        X[2*i] = np.zeros(d)
        X[2*i+1] = np.zeros((d,d+1))
    X[2*i+1] = np.zeros(d+1)
    X[2*L] = np.zeros(1)

    # layer 1
    X[0] = np.random.normal(loc=0.0, scale=1.0, size=d)

    # except last layer
    for i in range(1,2*L-1):
        if i%2==1:
            for j in range(0,d):
                for k in range(0,d+1):
                    X[i][j,k] = leaky_relu(sum(b[(i,j,k)] * np.append(X[i-1], At_list[i+1,j]) )) + np.random.normal(loc=0.0, scale=1.0)
        else:
            for j in range(0,d):
                X[i][j] =leaky_relu(sum(b[(i,j)]*X[i-1][j])) + np.random.normal(loc=0.0, scale=1.0)


    # last layer
    j=0
    for k in range(d+1):
        X[2*L-1][k] = leaky_relu(sum(b[(2*L-1,j,k)]* np.append(X[2*L-2], At_list[0]))) + np.random.normal(loc=0.0, scale=1.0)

    X[2*L][0] = leaky_relu(sum(bN*X[2*L-1] ))+ np.random.normal(loc=0.0, scale=1.0)

    return X


# for estimating mean
def generate_Xreward(At,hat_b,hat_bN,seed):
    """
    At_list: intervention
    hat_b: parameters except last layer
    hat_bN: parameters for last layer
    seed: random seeds
    output: reward
    """

    # Save the current state
    state = np.random.get_state()
    np.random.seed(seed)

    # store values
    X=[[] for i in range(2*L+1)]
    for i in range(L):
        X[2*i] = np.zeros(d)
        X[2*i+1] = np.zeros((d,d+1))
    X[2*i+1] = np.zeros(d+1)
    X[2*L] = np.zeros(1)

    # layer 1
    X[0] = np.random.normal(loc=0.0, scale=1.0, size=d)

    # except last layer
    for i in range(1,2*L-1):
        if i%2==1:
            for j in range(0,d):
                for k in range(0,d+1):
                    X[i][j,k] = leaky_relu(sum(hat_b[(i,j,k)] * np.append(X[i-1], At[i+1,j]) )) + np.random.normal(loc=0.0, scale=1.0)
        else:
            for j in range(0,d):
                X[i][j] =leaky_relu(sum(hat_b[(i,j)]*X[i-1][j])) + np.random.normal(loc=0.0, scale=1.0)


    #last two layers
    j=0
    for k in range(d+1):
        X[2*L-1][k] = leaky_relu(sum(hat_b[(2*L-1,j,k)]* np.append(X[2*L-2], At[0]))) + np.random.normal(loc=0.0, scale=1.0)

    X[2*L][0] = leaky_relu(sum(hat_bN*X[2*L-1] )) + np.random.normal(loc=0.0, scale=1.0)

    # Restore the previous state
    np.random.set_state(state)
    return X[2*L][0]



def estimate_calculate_UCB(b_TS, bN_TS,T=1000):
    """
    estimate UCBa(t)
    """
    UCB = np.full((2**((L-1)*d+1)), 0.)
    # for each arm calculate the UCB
    for action in range(2**((L-1)*d+1)):
        # get binary rep of action
        action_binary = [int(s) for s in "{0:b}".format(action)]
        action_binary[:0]=[0]*(((L-1)*d+1)-len(action_binary))

        AA = action_binary.copy()
        #construct At
        At_list = {}
        for y in range(2,2*L,2):
            for x in range(0, d):
                At_list[(y,x)] = AA.pop(0)*10-5
        At_list[0] = AA.pop(0)*10-5

        result = []
        for seed in range(T):
            result.append(generate_Xreward(At_list,b_TS,bN_TS,seed))
            # result.append(generate_X(At,b_TS,bN_TS))
        UCB[action] = np.mean(result)
    return UCB


UCB = estimate_calculate_UCB(b, bN,T=10000)
regret_action = -UCB + np.max(UCB)
# print(regret_action)



def GCB(N,T):
    """
    implementation of proposed GCB algorithm in hierarchical graph
    N: number of nodes
    T: time horizon
    """

    # initialize the estimators
    net = NN()

    hat_b={}
    for i in range(1,2*L-1):
        if i%2==1:
            for j in range(d):
                for k in range(d+1):
                    hat_b[(i,j,k)] =  np.random.uniform(-1, 1, d+1)
        else:
            for j in range(0,d):
                hat_b[(i,j)] = np.random.uniform(-1, 1, d+1)

    i=2*L-1
    j=0
    for k in range(0, d+1):
        hat_b[(i,j,k)] = np.random.uniform(-1, 1, d+1)

    hat_bN = np.random.uniform(-1, 1, d+1)

    hat_V={}
    for i in range(1,2*L-1):
        if i%2==1:
            for j in range(d):
                for k in range(d+1):
                    hat_V[(i,j,k)] = np.eye(d+1)
        else:
            for j in range(0,d):
                hat_V[(i,j)] = np.eye(d+1)
    i=2*L-1
    j=0
    for k in range(0, d+1):
        hat_V[(i,j,k)] = np.eye(d+1)
    hat_VN = np.eye(d+1)


    dataset = {}
    for i in range(1,2*L-1):
        if i%2==1:
            for j in range(d):
                for k in range(d+1):
                    dataset[(i,j,k)] = [[],[]]
        else:
            for j in range(0,d):
                dataset[(i,j)] = [[],[]]
    i=2*L-1
    j=0
    for k in range(0, d+1):
        dataset[(i,j,k)] = [[],[]]
    datasetN = [[],[]]


    # variables for saving the historical results
    regret = np.zeros(T)
    actions = np.zeros(T)
    
    # sequential decision 
    for t in range(T):
        print(t)
        if t%1000 == 0:
            print(t)
            sys.stdout.flush()
        # sample weights according to posterior
        b_TS = {}
        for i in range(1,2*L-1):
            if i%2==1:
                for j in range(d):
                    for k in range(d+1):
                        b_TS[(i,j,k)]  = np.random.multivariate_normal(hat_b[(i,j,k)].flatten(),np.linalg.inv(hat_V[i,j,k]))
            else:
                for j in range(0,d):
                    b_TS[(i,j)]  = np.random.multivariate_normal(hat_b[(i,j)].flatten(),np.linalg.inv(hat_V[i,j]))
        i=2*L-1
        j=0
        for k in range(0, d+1):
            b_TS[(i,j,k)] = np.random.multivariate_normal(hat_b[(i,j,k)].flatten(),np.linalg.inv(hat_V[i,j,k]))

        #sample the last layers
        bN_TS  = np.random.multivariate_normal(hat_bN.flatten(),np.linalg.inv(hat_VN))

        # store UCB for all actions
        UCB = estimate_calculate_UCB(b_TS, bN_TS)
 
        # choose action that maxmize the UCB
        At = np.argmax(UCB)
        # get binary representation of action
        At_binary = [int(s) for s in "{0:b}".format(At)]
        At_binary[:0]=[0]*(((L-1)*d+1)-len(At_binary))
        

        AA = At_binary.copy()
        #construct At
        At_list = {}
        for y in range(2,2*L,2):
            for x in range(0, d):
                At_list[(y,x)] = AA.pop(0)*10-5
        At_list[0] = AA.pop(0)*10-5

        

        # regret of action At
        regret[t] = regret_action[At]
        actions[t] = At
        # print(regret[t],actions[t])
        
        
        # generate possible corrupted samples
        X = generate_X(At_list)

        # get the training dataset
        for i in range(1,2*L-1):
            if i%2==1:
                for j in range(d):
                    for k in range(d+1):
                        dataset[(i,j,k)][0].append(np.append(X[i-1], At_list[i+1,j]))
                        dataset[(i,j,k)][1].append(X[i][j,k])
            else:
                for j in range(0,d):
                    dataset[(i,j)][0].append(X[i-1][j])
                    dataset[(i,j)][1].append(X[i][j])
        i=2*L-1
        j=0
        for k in range(0, d+1):
            dataset[(i,j,k)][0].append(np.append(X[2*L-2], At_list[0]))
            dataset[(i,j,k)][1].append(X[i][k])
        datasetN[0].append(X[2*L-1])
        datasetN[1].append(X[2*L][0])

        # update hat_V
        for i in range(1,2*L-1):
            if i%2==1:
                for j in range(d):
                    for k in range(d+1):
                        hat_V[(i,j,k)] += np.outer(np.append(X[i-1], At_list[i+1,j]), np.append(X[i-1], At_list[i+1,j]))
            else:
                for j in range(0,d):
                    hat_V[(i,j)] += np.outer(X[i-1][j],X[i-1][j])
        i=2*L-1
        j=0
        for k in range(0, d+1):
            hat_V[(i,j,k)] += np.outer(np.append(X[2*L-2], At_list[0]),np.append(X[2*L-2], At_list[0]))
        hat_VN += np.outer(X[2*L-1],X[2*L-1])




        # # update the parameters using gradient descent
        for i in range(1,2*L-1):
            if i%2==1:
                for j in range(d):
                    for k in range(d+1):
                        # get data
                        train_X = torch.tensor(dataset[(i,j,k)][0])
                        train_Y = torch.tensor(dataset[(i,j,k)][1])
                        # set weights
                        net.linear_one.weight.data = torch.tensor(hat_b[(i,j,k)])
                        optimizer = optim.SGD(net.parameters(), lr=0.01)
                        criterion = nn.MSELoss()
                        for _ in range(10):
                            optimizer.zero_grad()
                            if t>50:
                                random_indices = torch.randint(0, train_X.shape[0], (32,))
                                batch_X = train_X[random_indices]
                                batch_Y = train_Y[random_indices]
                                output = net(batch_X)
                                l2_loss = criterion(batch_Y, output)
                                l2_loss.backward()
                                optimizer.step()
                            else:
                                output = net(train_X)
                                # gradient decent
                                l2_loss = criterion(train_Y, output)
                                l2_loss.backward()
                                optimizer.step()
                        # get the results
                        hat_b[(i,j,k)] = net.linear_one.weight.data.numpy()
            else:
                for j in range(0,d):
                    # get data
                    train_X = torch.tensor(dataset[(i,j)][0])
                    train_Y = torch.tensor(dataset[(i,j)][1])
                    # set weights
                    net.linear_one.weight.data = torch.tensor(hat_b[(i,j)])
                    optimizer = optim.SGD(net.parameters(), lr=0.01)
                    criterion = nn.MSELoss()
                    for _ in range(10):
                        optimizer.zero_grad()
                        if t>50:
                            random_indices = torch.randint(0, train_X.shape[0], (32,))
                            batch_X = train_X[random_indices]
                            batch_Y = train_Y[random_indices]
                            output = net(batch_X)
                            l2_loss = criterion(batch_Y, output)
                            l2_loss.backward()
                            optimizer.step()
                        else:
                            output = net(train_X)
                            # gradient decent
                            l2_loss = criterion(train_Y, output)
                            l2_loss.backward()
                            optimizer.step()
                    # get the results
                    hat_b[(i,j)] = net.linear_one.weight.data.numpy()


        i=2*L-1
        j=0
        for k in range(0, d+1):
            train_X = torch.tensor(dataset[(i,j,k)][0])
            train_Y = torch.tensor(dataset[(i,j,k)][1])
            # set weights
            net.linear_one.weight.data = torch.tensor(hat_b[(i,j,k)])
            optimizer = optim.SGD(net.parameters(), lr=0.01)
            criterion = nn.MSELoss()
            for _ in range(10):
                optimizer.zero_grad()
                if t>50:
                    random_indices = torch.randint(0, train_X.shape[0], (32,))
                    batch_X = train_X[random_indices]
                    batch_Y = train_Y[random_indices]
                    output = net(batch_X)
                    l2_loss = criterion(batch_Y, output)
                    l2_loss.backward()
                    optimizer.step()
                else:
                    output = net(train_X)
                    # gradient decent

                    l2_loss = criterion(train_Y, output)
                    l2_loss.backward()
                    optimizer.step()
            # get the results
            hat_b[(i,j,k)] = net.linear_one.weight.data.numpy()

        train_X = torch.tensor(datasetN[0])
        train_Y = torch.tensor(datasetN[1])
        # set weights
        net.linear_one.weight.data = torch.tensor(hat_bN)
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        for _ in range(10):
            optimizer.zero_grad()
            if t>50:
                random_indices = torch.randint(0, train_X.shape[0], (32,))
                batch_X = train_X[random_indices]
                batch_Y = train_Y[random_indices]
                output = net(batch_X)
                l2_loss = criterion(batch_Y, output)
                l2_loss.backward()
                optimizer.step()    
            else:        
                output = net(train_X)
                # gradient decent
                l2_loss = criterion(train_Y, output)
                l2_loss.backward()
                # print(net.linear_one.weight.grad)
                optimizer.step()
        # get the results
        hat_bN = net.linear_one.weight.data.numpy()


            
    return regret, hat_b, hat_bN, actions, UCB




avgRegret_GCB=np.zeros(T)


#runing the trail
iteration=1
for i in range(iteration):
    t1=time.time()
    print(i)
    regret_GCB, _, _, _, _= GCB(N, T)
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
# plt.plot(np.cumsum(avgRegret_GCB),linewidth=3)
# plt.grid()
# plt.ylabel('Cumulated Regret')
# plt.xlabel('Number of Iterations')
# plt.legend(["GCB"],borderpad=0.2)
# plt.savefig('./result/plot/regret_TS_NN_d%dL%d_%d.pdf'%(d,L,seed),bbox_inches='tight',dpi=200)
# plt.close()
        
np.save('./result/data/regret_TS_NN_d%dL%d_%d'%(d,L,seed), avgRegret_GCB)


