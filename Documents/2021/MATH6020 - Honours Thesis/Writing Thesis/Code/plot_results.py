import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import fishery_model_functions as f
import re

parser = argparse.ArgumentParser(description='read pomdpx file',
            formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-c', type=int, action="store", dest='c', default=5000)
args = parser.parse_args()

cost = args.c
actions = []
states = []
observations = []
observation_probability = []
reward = []

transitionworld = np.load('transition_world_model.npy')

obs_probs = np.load('obs_prob_fishery_model.npy')
transition = np.load('transition_fishery_model.npy')
biomass_vector = np.load('biomass_vector_fishery_model.npy')
action_vector = np.load('action_vector_fishery_model.npy')
K_vector = np.load('K_vector_fishery_model.npy')
r_vector = np.load('r_vector_fishery_model.npy')
states_vector = [(list(biomass),action,capacity,proliferation) for biomass in biomass_vector 
                                                         for action in action_vector
                                                         for capacity in K_vector
                                                         for proliferation in r_vector]
observations_vector = [(list(biomass),list(catch),capacity,proliferation) for biomass in biomass_vector 
                                                         for catch in biomass_vector
                                                         for capacity in K_vector
                                                         for proliferation in r_vector]

Kmax = biomass_vector[-1][1]

def plot_fishery_transition_matrix(a,K,rho):
    indices = []
    axis_labels = []
    for i,state in enumerate(states_vector):
        if state[2] == K and state[3] == rho and state[1] == 0.4:
            indices.append(i)
            axis_labels.append(str(state[0]))
    rows_removed = transition[a][indices]
    cols_removed = np.transpose(rows_removed)[indices]
    p = plt.figure()
    p = sns.heatmap(np.transpose(cols_removed), cmap = "flare",cbar_kws={'label': 'Transition Probability'})
    p.set_xticks(np.array(range(len(axis_labels)))+0.5)
    p.set_yticks(np.array(range(len(axis_labels)))+0.5)
    p.set_xticklabels(axis_labels,rotation = 45)
    p.set_yticklabels(axis_labels,rotation = 0)
    plt.xlabel('Next Biomass')
    plt.ylabel('Current Biomass')
    
    
for a in range(len(action_vector)):
    K = K_vector[-1]
    r = r_vector[-1]
    plot_fishery_transition_matrix(a,K,r) 
    plt.savefig('transitionfishery_%d.png' % a)
    
def plot_observation_matrix(a,K,rho):
    state_indices = []
    y_axis_labels = []
    obs_indices = []
    x_axis_labels = []
    for i,state in enumerate(states_vector):
        if state[2] == K and state[3] == rho:
            state_indices.append(i)
            y_axis_labels.append(str(state[0:2]))
    for j,obs in enumerate(observations_vector):
        if obs[2] == K and obs[3] == rho:
            obs_indices.append(j)
            x_axis_labels.append(str(obs[0:2]))        
    rows_removed = obs_probs[a][state_indices]
    cols_removed = np.transpose(rows_removed)[obs_indices]
    p = plt.figure(a)
    p = sns.heatmap(np.transpose(cols_removed))
    p.set_xticks(range(len(x_axis_labels)))
    p.set_yticks(range(len(y_axis_labels)))
    p.set_xticklabels(x_axis_labels,rotation = 90)
    p.set_yticklabels(y_axis_labels,rotation = 0)
    if a == len(transition)-1:
        plt.title('Observations for Action: No Stock Assessment')
    else:
        plt.title('Observations for Action: {}'.format(action_vector[a]))
    plt.xlabel('Observation')
    plt.ylabel('State')
    
# Specify location of DESPOT output file
with open('C:\cygwin64\home\Montana\despot\examples\pomdpx_models\doublemodel_output.txt','r') as file:
    rounds = [line for line in file.readlines() if line.startswith('#####')]
    num_runs = len(rounds)
    actions = [[] for i in range(num_runs)]
    biomass = [[] for i in range(num_runs)]
    states_index = [[] for i in range(num_runs)]
    obs_catch = [[] for i in range(num_runs)]
    obs_B = [[] for i in range(num_runs)]
    obs_K = [[] for i in range(num_runs)]
    obs_r = [[] for i in range(num_runs)]
    observations = [[] for i in range(num_runs)]
    observation_probability = [[] for i in range(num_runs)]
    reward = [[] for i in range(num_runs)] 
    

with open('C:\cygwin64\home\Montana\despot\examples\pomdpx_models\doublemodel_output.txt','r') as file:
    run = -1
    for i, line in enumerate(file.readlines()):
        if line.startswith('####'):
            run += 1
        if line.startswith('- Action'):
            actions[run].append(int(re.findall(r'[0-9]',line)[0]))
        if line.startswith('[b-h_1'):
            try:
                states_index[run].append(int(re.findall(r'[0-9]\d+',line)[0]))
            except:
                  states_index[run].append(int(re.findall(r'[0-9]',line)[0]))              
        if line.startswith('  discounted / undiscounted ='):
            try:
                reward[run].append(int(re.findall(r'-?[0-9]\d+', line)[0]))
            except:
                reward[run].append(int(re.findall(r'-?[0-9]', line)[0]))
        if line.startswith('- Observation'):
            observations[run].append(int(re.findall(r'\d+', line)[0]))
            obs_catch[run].append(observations_vector[int(re.findall(r'\d+', line)[0])][0])
            obs_B[run].append(observations_vector[int(re.findall(r'\d+', line)[0])][1])
            obs_K[run].append(observations_vector[int(re.findall(r'\d+', line)[0])][2])
            obs_r[run].append(observations_vector[int(re.findall(r'\d+', line)[0])][3])
                
num_steps = 90
# Plot actions
fig, ax = plt.subplots()
for i in range(num_runs):
    p = plt.plot(range(num_steps),action_vector[actions[i]], 'o', color = '#191970')
plt.ylim(0,1)
plt.title('Optimal Policy')
ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
ax.set_yticklabels([0,0.2,0.4,0.6,0.8,'NSA'])
plt.ylabel('Harvest')
plt.xlabel('Time')
plt.savefig('actions_%d.png' % cost)

#Plot reward
fig = plt.figure()
for i in range(num_runs):
    plt.plot(range(num_steps),reward[i], 'o', color = '#DC143C')
plt.title('Cumulative Discounted Reward')
plt.ylabel('Reward')
plt.xlabel('Time')
plt.legend(['run1','run2'])
plt.savefig('reward_%d.png' % cost)

# Plot biomass
fig = plt.figure()
for i in range(num_runs):
    for index in states_index[i]:
        biomass[i].append(states_vector[index][0])
    plt.plot(range(num_steps+1),np.mean(biomass[i],1), '-o')
plt.ylim(0,Kmax)
plt.title('Biomass from POMDP output')
plt.ylabel('Biomass')
plt.xlabel('Time')
plt.legend(['run1','run2'])
plt.savefig('statebiomass_%d.png' % cost)

# Plot observations
fig = plt.figure()
for i in range(num_runs):
    plt.plot(range(num_steps),np.mean(obs_catch[i],1), '-o')
plt.ylim(0,Kmax)
plt.title('Observed Catch from POMDP output')
plt.ylabel('Catch')
plt.xlabel('Time')
plt.legend(['run1','run2'])
plt.savefig('observedcatch_%d.png' % cost)

for i in range(num_runs):
    fig = plt.figure()
    fig, axs = plt.subplots(2,2,sharex=True)
    axs[0,0].plot(range(num_steps),np.mean(obs_catch[i],1), '-o', color='#228B22')
    axs[0,1].plot(range(num_steps),np.mean(obs_B[i],1), '-o', color = '#8B008B')
    axs[1,0].plot(range(num_steps),obs_K[i], '-o', color = '#F4A460')
    axs[1,1].plot(range(num_steps),obs_r[i], '-o', color = '#6495ED')
    axs[0,0].set(ylabel='Catch')
    axs[0,0].set_ylim([0,Kmax])
    axs[0,1].set_ylim([0,Kmax])
    axs[0,1].yaxis.set_label_position("right")
    axs[0,1].yaxis.tick_right()
    axs[0,1].set(ylabel='Biomass')
    axs[1,0].set(ylabel='K',xlabel='Time')
    axs[1,1].set(ylabel='ρ',xlabel='Time')
    axs[1,1].yaxis.set_label_position("right")
    axs[1,1].yaxis.tick_right()
fig.savefig('observations_%d.png' % cost)


# belief: 1st run only
belief = np.zeros((num_steps+1,len(states_vector)))
belief[0] = f.initial_belief(states_vector)


#Initial Belief Plot
fig, ax = plt.subplots()
ax.plot(belief[0],'o',label='Initial Belief', color ='#008080')
fig.suptitle('Initial Belief State (pmf)')
ax.set(xlabel='State', ylabel='Probability')
plt.text(25,0.0088,'e.g. ([0, 2000], 0.4, 10000.0, 2.5)')
plt.text(len(states_vector)/2,0.0005,'e.g. ([4000, 6000], 1.0, 0.0, 0.0)')



for step in range(0,num_steps):
    a = actions[0][step]
    o = observations[0][step]
    for s_dash in range(len(states_vector)):
        sum_term = 0
        for s in range(len(states_vector)):
            sum_term += transition[a][s][s_dash] * belief[step][s]
        belief[step+1][s_dash] = obs_probs[a][s_dash][o] * sum_term     
    belief[step+1] = belief[step+1]/sum(belief[step+1])

# Plot belief distribution        
fig = plt.figure()
plt.yscale('symlog', linthresh=0.015)
plt.plot(belief[0],'o',label='t=0', color = '#008080', alpha = 0.7, ms = 5)
plt.title('Belief States')
plt.xlabel('State')
plt.ylabel('Probability')
plt.plot(belief[1],'o',label='t=1', alpha = 0.7, ms = 5)
plt.plot(belief[2],'o',label='t=2', color = '#FFD700', alpha = 0.7, ms = 5)
plt.plot(belief[last_working_step],'o',label='t=90', color = '#D2691E', alpha = 0.7, ms =5)
plt.legend()

number_list = list(belief[last_working_step])
max_value = max(number_list)
max_index = number_list.index(max_value)
print('State with highest belief probability:',states_vector[max_index])
fig.savefig('Belief%d.png' % cost)

