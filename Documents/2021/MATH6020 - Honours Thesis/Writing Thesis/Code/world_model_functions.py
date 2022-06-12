import numpy as np
import scipy.stats as ss
import sys

def BevertonHolt(x,K,rho,a,noise):
    return (rho * K * (1.0 - a) * x) / (K + (rho - 1.0) * x)*np.exp(np.random.normal(0, noise,size=1))

def discrete_BH_transition(states_vector, action_vector, r, K, noise=0, N=10):
    N_a, N_S = len(action_vector), len(states_vector)
    transition = np.zeros((N_a, N_S, N_S))
    for i in range(N_a):
        if action_vector[i] == 1:
            for j in range(N_S):
                a = states_vector[j][1]
                ss0 = np.random.uniform(states_vector[j][0][0], states_vector[j][0][1], size=N)
                rho = np.random.normal(r, 0.1, size=N)
                ss1 = BevertonHolt(ss0, K, rho, a, noise=noise)*np.exp(np.random.normal(0.0,0.2,size=N))
                for m in range(len(ss1)):
                    if ss1[m] <0:
                        ss1[m] = 0
                    if ss1[m]>K:
                        ss1[m] = K
                for k in range(N_S):
                    if states_vector[k][1] == a and states_vector[k][2] == K and states_vector[k][3] == r:
                        sk = states_vector[k][0]
                        l, u = sk
                        if np.abs(sk[1] - K) < 1E-10:
                            u += 1E-10
                        transition[i][j][k] = ((ss1 >= l) & (ss1 < u)).mean()
                transition[i][j] = transition[i][j]/sum(transition[i][j])
        else:
            a = action_vector[i]
            for j in range(N_S):
                ss0 = np.random.uniform(states_vector[j][0][0], states_vector[j][0][1], size=N)
                rho = np.random.normal(r, 0.1, size=N)
                ss1 = BevertonHolt(ss0, K, rho, a, noise=noise)*np.exp(np.random.normal(0.0,0.1,size=N))
                for m in range(len(ss1)):
                    if ss1[m] <0:
                        ss1[m] = 0
                    if ss1[m]>K:
                        ss1[m] = K
                for k in range(N_S):
                    if states_vector[k][1] == a and states_vector[k][2] == K and states_vector[k][3] == r:
                        sk = states_vector[k][0]
                        l, u = sk
                        if np.abs(sk[1] - K) < 1E-10:
                            u += 1E-10
                        transition[i][j][k] = ((ss1 >= l) & (ss1 < u)).mean()
                if sum(transition[i][j]) != 1:
                    transition[i][j] = transition[i][j]/sum(transition[i][j])
    return transition

def reward_function(action, state, cost_of_stock_assessment, profit):
    if action == 1: # No Stock Assessment
        penalty = 0
        reward = profit*state[1]*np.mean(state[0])
    else:
        penalty = cost_of_stock_assessment
        reward = profit*state[1]*np.mean(state[0])
    return reward - penalty

def biomass_pdf(B,Bstar,sigma1,N=100):
    sample = np.random.normal(Bstar,sigma1,N)
    return sum((sample >= B[0]) & (sample <= B[1]))/N
    
def observation_function(states_vector, action_vector, observations_vector, Rstar, Kstar, Bstar, N = 10):
    sigma1 = 1000
    sigma2 = 1000   
    sigma3 = 0.1
    N_a, N_S, N_O = len(action_vector), len(states_vector), len(observations_vector)
    obs_transition = np.zeros((N_a, N_S, N_O))
    for i in range(N_a):
        print('\n',i, '/', N_a-1)
        for j in range(N_S):
            progress = (j+1)/N_S
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*progress), 100*progress))
            sys.stdout.flush()
            if action_vector[i] == 1:
                for k in range(N_O):
                    action = states_vector[j][1]
                    B = observations_vector[k][1]
                    K = observations_vector[k][2]
                    R = observations_vector[k][3]
                    ss0 = np.random.uniform(B[0],B[1],N)
                    catch = action*ss0*np.exp(np.random.normal(0.0,0.4,size=N))
                    l,u = observations_vector[k][0]
                    obs_transition[i][j][k] = sum((catch >= l) & (catch <= u))/N
                obs_transition[i][j] = obs_transition[i][j]/sum(obs_transition[i][j])
            else:
                for k in range(N_O):
                    action = action_vector[i]
                    B = observations_vector[k][1]
                    K = observations_vector[k][2]
                    R = observations_vector[k][3]
                    ss0 = np.random.uniform(B[0],B[1],N)
                    catch = action*ss0*np.exp(np.random.normal(0.0,0.4,size=N))
                    l,u = observations_vector[k][0]
                    LHS = sum((catch >= l) & (catch <= u))/N
                    RHS = biomass_pdf(B,Bstar,sigma1)*ss.norm.pdf(K,Kstar,sigma2)*ss.norm.pdf(R,Rstar,sigma3)
                    obs_transition[i][j][k] = LHS*RHS
                obs_transition[i][j] = obs_transition[i][j]/sum(obs_transition[i][j])
    return obs_transition


def initial_belief(states):
    belief = np.ones(len(states))
    for i, state in enumerate(states):
        if state[0][1] > state[2]:
            belief[i] = 0
        if state[1] == 1.0:
            belief[i] = 0
        if state[2] != 300000:
            belief[i] = 0
        if state[3] != 1.0:
            belief[i] = 0
    return belief / sum(belief)

