import random
import numpy as np
import matplotlib.pyplot as plt
import discrete_pendulum
from numpy import save
from numpy import load


# Constant params
gamma = 0.95 # discount factor
eps = 0.2 # epsilon-greedy probability param (For SARSA and Q-Learning)
# n_episodes = 10000 # number of episodes (For SARSA and Q-Learning)
alpha = 1/n_episodes # learning rate
alpha = 0.7
experiment = False # change to 'True' to generate Learning curves for different alpha and epsilon values

# Experiment params
min_alpha = 0.1
max_alpha = 0.8
n_alpha = 4
min_eps = 0.001
max_eps = 0.2
n_eps = 4



# Create environment
#   By default, both the state space (theta, thetadot) and the action space
#   (tau) are discretized with 31 grid points in each dimension, for a total
#   of 31 x 31 states and 31 actions.
#   Note that there will only be a grid point at "0" along a given dimension
#   if the number of grid points in that dimension is odd.    
# env = discrete_pendulum.Pendulum(n_theta=15, n_thetadot=21)
# env = discrete_pendulum.Pendulum(n_theta=30, n_thetadot=36)
env = discrete_pendulum.Pendulum(n_theta=11, n_thetadot=51, n_tau=21)



################# SARSA #################
def sarsa(gamma,eps,n_episodes,alpha,experiment):
    # initialization
    # Q = np.ones((env.num_states, env.num_actions))
    Q = load('Q_sarsa.npy')
    print('State and control dimesnions: ')
    print(np.shape(Q))
    episodes = []
    pi = np.random.randint(0,env.num_actions,env.num_states)
    # pi = load('pi_sarsa.npy')
    Gs = []

    for i in range(0,n_episodes+1,1): # new trajectory/episode
        if n_episodes>1000 and np.mod(i,round(n_episodes/100)) == 0:    
            print(i)
        if n_episodes<=1000:
            print(i)
        episodes.append(i)
        G = 0
        power = 0
        r = 0
        s = env.reset()

        rand_n = np.random.uniform(0, 1, 1)
        if rand_n < eps:
            a = np.random.randint(0, env.num_actions, 1).item()
        else:
            a = np.argmax(Q[s,:])

        done = False
        while not done:
            (s1, r, done) = env.step(a)
            
            rand_n = np.random.uniform(0, 1, 1)
            if rand_n < eps:
                a1 = np.random.randint(0, env.num_actions, 1).item()
            else:
                a1 = np.argmax(Q[s1,:])
            Q[s,a] = Q[s,a] + alpha*(r + gamma*Q[s1,a1] - Q[s,a]) 

            s = s1
            a = a1

            G += (gamma**power)*r
            power += 1
        Gs.append(G)            
    for j in range(0,env.num_states,1):
        pi[j] = np.argmax(Q[j,:])
    # print(pi)

    if experiment == False:
        s = env.reset()
        log = {
            't': [0],
            's': [s],
            'a': [],
            'r': [],
            'theta': [env.x[0]],        # agent does not have access to this, but helpful for display
            'thetadot': [env.x[1]],     # agent does not have access to this, but helpful for display
        }
        # Simulate until episode is done
        done = False
        while not done:
            a = random.randrange(env.num_actions)
            (s, r, done) = env.step(a)
            log['t'].append(log['t'][-1] + 1)
            log['s'].append(s)
            log['a'].append(a)
            log['r'].append(r)
            log['theta'].append(((env.x[0] + np.pi) % (2 * np.pi)) - np.pi)
            log['thetadot'].append(env.x[1])

        # Plot data and save to png file
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        ax[0].plot(log['t'], log['s'])
        ax[0].plot(log['t'][:-1], log['a'])
        ax[0].plot(log['t'][:-1], log['r'])
        ax[0].legend(['s', 'a', 'r'])
        ax[1].plot(log['t'], log['theta'])
        ax[1].plot(log['t'], log['thetadot'])
        ax[1].legend(['theta', 'thetadot'])
        fig.suptitle('State, Action, and Reward Trajectories - SARSA')
        plt.savefig('figures/pendulum/sarsa_traj_pendulum.png')

        # Plot the learning curve
        plt.figure(plt.gcf().number+1)
        plt.plot(episodes,Gs)
        plt.title('SARSA Learning Curve')
        plt.xlabel('number of episodes')
        plt.ylabel('G')
        plt.savefig('figures/pendulum/sarsa_g_pendulum.png')
    return(Q,pi,Gs,episodes,n_episodes)
##############################################



################# Q-LEARNING #################

def Qlearning(gamma,eps,n_episodes,alpha,experiment):
    Q = np.ones((env.num_states, env.num_actions))
    # Q = load('Q_qlearn.npy')
    print('State and control dimesnions: ')
    print(np.shape(Q))
    Q_candidates = np.zeros(env.num_actions)
    episodes = []
    pi = np.random.randint(0,env.num_actions,env.num_states)
    # pi = load('pi_qlearn.npy')
    Gs = []

    for i in range(0,n_episodes+1,1): # new trajectory/episode
        if n_episodes>1000 and np.mod(i,round(n_episodes/100)) == 0:    
            print(i)
        if n_episodes<=1000:
            print(i)
        episodes.append(i)
        G = 0
        power = 0
        r = 0
        s = env.reset()
        done = False
        while not done:
            rand_n = np.random.uniform(0, 1, 1)
            if rand_n < eps:
                a = np.random.randint(0, env.num_actions, 1).item()
            else:
                a = np.argmax(Q[s,:])
            (s1, r, done) = env.step(a)
            for a_o in range(0,env.num_actions,1):
                Q_candidates[a_o] = Q[s,a] + alpha*(r + gamma*Q[s1,a_o] - Q[s,a]) 
            Q[s,a] = np.max(Q_candidates)
            s = s1

            G += (gamma**power)*r
            power += 1
        Gs.append(G)            
    for j in range(0,env.num_states,1):
        pi[j] = np.argmax(Q[j,:])
    # print(pi)

    if experiment == False:
        s = env.reset()
        log = {
            't': [0],
            's': [s],
            'a': [],
            'r': [],
            'theta': [env.x[0]],        # agent does not have access to this, but helpful for display
            'thetadot': [env.x[1]],     # agent does not have access to this, but helpful for display
        }
        # Simulate until episode is done
        done = False
        while not done:
            a = random.randrange(env.num_actions)
            (s, r, done) = env.step(a)
            log['t'].append(log['t'][-1] + 1)
            log['s'].append(s)
            log['a'].append(a)
            log['r'].append(r)
            log['theta'].append(((env.x[0] + np.pi) % (2 * np.pi)) - np.pi)
            log['thetadot'].append(env.x[1])

        # Plot data and save to png file
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        ax[0].plot(log['t'], log['s'])
        ax[0].plot(log['t'][:-1], log['a'])
        ax[0].plot(log['t'][:-1], log['r'])
        ax[0].legend(['s', 'a', 'r'])
        ax[1].plot(log['t'], log['theta'])
        ax[1].plot(log['t'], log['thetadot'])
        ax[1].legend(['theta', 'thetadot'])
        fig.suptitle('State, Action, and Reward Trajectories - Q-learning')
        plt.savefig('figures/pendulum/qlearn_traj_pendulum.png')
        
        # Plot the learning curve
        plt.figure(plt.gcf().number+1)
        plt.plot(episodes,Gs)
        plt.title('Q-Learning Learning Curve')
        plt.xlabel('number of episodes')
        plt.ylabel('G')
        plt.savefig('figures/pendulum/qlearn_g_pendulum.png')
    return(Q,pi,Gs,episodes,n_episodes)
####################################################



################# TD(0) ESTIMATION #################
def  TD0(gamma,eps,n_episodes,alpha,experiment):
  # initialization
    V = (1/env.num_states)*np.ones(env.num_states)
    V_q = (1/env.num_states)*np.ones(env.num_states)
    
    # Estimatting SARSA's Value Function
    Q,pi,Gs,episodes,n_episodes = sarsa(gamma,eps,n_episodes,alpha,experiment)
    save('Q_sarsa.npy', Q)
    save('pi_sarsa.npy', pi)
    alpha = 1/n_episodes

    for i in range(0,n_episodes+1,1):
        s = env.reset()
        done = False
        while not done:
            a = pi[s]
            (s1, r, done) = env.step(a)
            V[s] = V[s] + alpha*(r + gamma*V[s1] - V[s])
            s = s1
    # Plot the learning curve
    plt.figure(plt.gcf().number+1)
    plt.plot(range(0,env.num_states,1),V)
    plt.title('SARSA Value Function Learned by TD(0)')
    plt.xlabel('state s')
    plt.ylabel('V')
    plt.savefig('figures/pendulum/sarsa_v_td0_gridworld.png')

    plt.figure(plt.gcf().number+1)
    plt.plot(range(0,env.num_states,1),pi)
    plt.title('SARSA Optimal Policy')
    plt.xlabel('state s')
    plt.ylabel('\pi')
    plt.savefig('figures/pendulum/sarsa_pi_td0_gridworld.png')
    
    # Estimatting Q-learning's Value Function
    Q_q,pi_q,Gs_q,episodes,n_episodes = Qlearning(gamma,eps,n_episodes,alpha,experiment)
    save('Q_qlearn.npy', Q)
    save('pi_qlearn.npy', pi)        
    alpha = 1/n_episodes

    for i in range(0,n_episodes+1,1):
        s = env.reset()
        done = False
        while not done:
            a = pi_q[s]
            (s1, r, done) = env.step(a)
            V_q[s] = V_q[s] + alpha*(r + gamma*V_q[s1] - V_q[s])
            s = s1

        # Plot the learning curve
    plt.figure(plt.gcf().number+1)
    plt.plot(range(0,env.num_states,1),V_q)
    plt.title('Q-Learning Value Function Learned by TD(0)')
    plt.xlabel('state s')
    plt.ylabel('V')
    plt.savefig('figures/pendulum/qlearn_v_td0_gridworld.png')

    plt.figure(plt.gcf().number+1)
    plt.plot(range(0,env.num_states,1),pi_q)
    plt.title('Q-Learning Optimal Policy')
    plt.xlabel('state s')
    plt.ylabel('\pi')
    plt.savefig('figures/pendulum/qlearn_pi_td0_gridworld.png')

    print('The Estimated Value Functions (using SARSA and Q-Learning computed policies) are:')
    return V, V_q
if experiment == False:
    TD0(gamma,eps,n_episodes,alpha,experiment) 
#######################################################



################## Alpha Experiments ##################
def plot_alphas_sarsa(gamma,eps,n_episodes,alpha,experiment,min_alpha,max_alpha,n_alpha):
    alpha_array = np.linspace(min_alpha,max_alpha,n_alpha)
    print(np.size(alpha_array))
    plt.figure(plt.gcf().number+1)
    for alpha_index in range(0,np.size(alpha_array),1):
        alpha = alpha_array[alpha_index]
        Q,pi,Gs,episodes,n_episodes = sarsa(gamma,eps,n_episodes,alpha,experiment)
        # Plot the learning curve
        plt.plot(episodes,Gs,label='%s ' % alpha)
        plt.title('SARSA Learning Curves for different Alphas')
        plt.xlabel('number of episodes')
        plt.ylabel('G')
    plt.legend()    
    plt.savefig('figures/pendulum/exp_alpha_sarsa_g_gridworld.png')
if experiment == True:
    plot_alphas_sarsa(gamma,eps,n_episodes,alpha,experiment,min_alpha,max_alpha,n_alpha)

def plot_alphas_qlearn(gamma,eps,n_episodes,alpha,experiment,min_alpha,max_alpha,n_alpha):
    alpha_array = np.linspace(min_alpha,max_alpha,n_alpha)
    print(np.size(alpha_array))
    plt.figure(plt.gcf().number+1)
    for alpha_index in range(0,np.size(alpha_array),1):
        alpha = alpha_array[alpha_index]
        Q,pi,Gs,episodes,n_episodes = Qlearning(gamma,eps,n_episodes,alpha,experiment)
        # Plot the learning curve
        plt.plot(episodes,Gs,label='%s ' % alpha)
        plt.title('Q-Learning Learning Curves for different Alphas')
        plt.xlabel('number of episodes')
        plt.ylabel('G')
    plt.legend()    
    plt.savefig('figures/pendulum/exp_alpha_qlearn_g_gridworld.png')
if experiment == True:
    plot_alphas_qlearn(gamma,eps,n_episodes,alpha,experiment,min_alpha,max_alpha,n_alpha)
#######################################################



################# Epsilon Experiments #################
def plot_epsilons_sarsa(gamma,eps,n_episodes,alpha,experiment,min_eps,max_eps,n_eps):
    eps_array = np.linspace(min_eps,max_eps,n_eps)
    print(np.size(eps_array))
    plt.figure(plt.gcf().number+1)
    for eps_index in range(0,np.size(eps_array),1):
        eps = eps_array[eps_index]
        Q,pi,Gs,episodes,n_episodes = sarsa(gamma,eps,n_episodes,alpha,experiment)
        # Plot the learning curve
        plt.plot(episodes,Gs,label='%s ' % eps)
        plt.title('SARSA Learning Curves for different Epsilons')
        plt.xlabel('number of episodes')
        plt.ylabel('G')
    plt.legend()    
    plt.savefig('figures/pendulum/exp_eps_sarsa_g_gridworld.png')
if experiment == True:
    plot_epsilons_sarsa(gamma,eps,n_episodes,alpha,experiment,min_eps,max_eps,n_eps)

def plot_epsilons_qlearn(gamma,eps,n_episodes,alpha,experiment,min_eps,max_eps,n_eps):
    eps_array = np.linspace(min_eps,max_eps,n_eps)
    print(np.size(eps_array))
    plt.figure(plt.gcf().number+1)
    for eps_index in range(0,np.size(eps_array),1):
        eps = eps_array[eps_index]
        Q,pi,Gs,episodes,n_episodes = Qlearning(gamma,eps,n_episodes,alpha,experiment)
        # Plot the learning curve
        plt.plot(episodes,Gs,label='%s ' % eps)
        plt.title('Q-Learning Learning Curves for different Epsilons')
        plt.xlabel('number of episodes')
        plt.ylabel('G')
    plt.legend()    
    plt.savefig('figures/pendulum/exp_eps_qlearn_g_gridworld.png')
if experiment == True:
    plot_epsilons_qlearn(gamma,eps,n_episodes,alpha,experiment,min_eps,max_eps,n_eps)
#####################################################