import random
import numpy as np
import matplotlib.pyplot as plt
import statistics
import gridworld

# Create environment
env = gridworld.GridWorld(hard_version=False)

# Constant params
gamma = 0.95 # discount factor
threshold = 1e-3 # fixed point tolerance (for Policy and Value Iterations)
eps = 0.05 # epsilon-greedy probability param (For SARSA and Q-Learning)
n_episodes = 1000 # number of episodes (For SARSA and Q-Learning)
alpha = 1/n_episodes # learning rate
experiment = False # change to 'True' to generate Learning curves for different alpha and epsilon values

# Experiment params
min_alpha = 0.1
max_alpha = 0.8
n_alpha = 4
min_eps = 0.001
max_eps = 0.2
n_eps = 4



############ VALUE ITERATION ############
def value_iteration(gamma,threshold):
    # initialization
    V = np.zeros(env.num_states)
    pi = np.zeros(env.num_states)
    V_sum = np.zeros(env.num_actions)
    V_sum_s1 = np.zeros(env.num_states)
    mean_V = []
    error = 1

    i = 0 # fixed point map iterations

    while threshold<error:
        error = 0
        i += 1
        mean_V.append(0)
        for s in range(0,env.num_states,1):
            V_old = V[s]
            for a in range(0,env.num_actions,1):
                for s1 in range(0,env.num_states,1):
                    V_sum_s1[s1] = env.p(s1, s, a)*(env.r(s,a)+gamma*V[s1])
                V_sum[a] = np.sum(V_sum_s1)     
            V[s] = np.max(V_sum)
            pi[s] = np.argmax(V_sum)
            error = max(error,np.abs(V_old-V[s]))
            mean_V[i-1] = statistics.mean(V)
    print('Final Value Iteration error: ') 
    print(error)        
    # Plot value function data and save to png file
    plt.figure(plt.gcf().number+1)
    plt.plot(range(0,i,1), mean_V)
    plt.title('Value Iteration Learning Curve')
    plt.xlabel('iterations')
    plt.ylabel('Mean V')
    plt.savefig('figures/gridworld/Viteration_meanV_gridworld.png')

    ## Plotting the agent's trajectories
    s = env.reset() # choose an initial state randomly
    # Create log to store data from simulation
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }
    done = False
    while not done:
        a = pi[s]
        (s, r, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)
    # Plot the agent data and save to png file
    plt.figure(plt.gcf().number+1)
    plt.plot(log['t'], log['s'])
    plt.plot(log['t'][:-1], log['a'])
    plt.plot(log['t'][:-1], log['r'])
    plt.title('State, Action, and Reward Trajectories - Value Iteration')
    plt.xlabel('time step')
    plt.legend(['s', 'a', 'r'])
    plt.savefig('figures/gridworld/Viteration_traj_gridworld.png')
if experiment == False:
    value_iteration(gamma,threshold)
##########################################



############ POLICY ITERATION ############
def policy_iteration(gamma,threshold):
    # initialization
    V = np.zeros(env.num_states)
    pi = np.random.randint(0,5,1)*np.ones(env.num_states)
    V_sum_s1 = np.zeros(env.num_states)
    Vp_sum_s1 = np.zeros(env.num_states)
    Vp_a = np.zeros(env.num_actions)
    old_pi = np.zeros(env.num_states)
    mean_V = []
    error = 1
    policy_stable = False

    i = 0 # Policy iterations

    while policy_stable == False:
        while threshold<error:
            i +=1
            error = 0
            for s in range(0,env.num_states,1):
                V_old = V[s]
                a = pi[s]
                for s1 in range(0,env.num_states,1):
                    V_sum_s1[s1] = env.p(s1, s, a)*(env.r(s, a)+gamma*V[s1])
                V[s] = np.sum(V_sum_s1)   
                error = max(error,np.abs(V_old-V[s]))
            # print(error)
            mean_V.append(statistics.mean(V))


        for s in range(0,env.num_states,1):
            old_pi[s] = pi[s]  
        policy_stable = True
        for s in range(0,env.num_states,1):
            for a in range(0,env.num_actions,1):
                for s1 in range(0,env.num_states,1):
                    Vp_sum_s1[s1] = env.p(s1, s, a)*(env.r(s, a)+gamma*V[s1])
                Vp_a[a] = np.sum(Vp_sum_s1)
            pi[s] = np.argmax(Vp_a)
        # print(pi)
        if np.linalg.norm(abs(old_pi-pi),np.inf) > 0:
            error = 1
            policy_stable = False
        if policy_stable == True:
            break
    # print(i)

    ## Plotting the agent's trajectories
    s = env.reset() # choose an initial state randomly
    # Create log to store data from simulation
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }
    done = False
    while not done:
        a = pi[s]
        (s, r, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)
    # Plot the agent data and save to png file
    plt.figure(plt.gcf().number+1)
    plt.plot(log['t'], log['s'])
    plt.plot(log['t'][:-1], log['a'])
    plt.plot(log['t'][:-1], log['r'])
    plt.title('State, Action, and Reward Trajectories - Policy Iteration')
    plt.xlabel('time step')
    plt.legend(['s', 'a', 'r'])
    plt.savefig('figures/gridworld/Piteration_traj_gridworld.png')
    
    # Plot the learning curve
    plt.figure(plt.gcf().number+1)
    plt.plot(range(0,i,1), mean_V)
    plt.title('Policy Iteration Learning Curve')
    plt.xlabel('iterations')
    plt.ylabel('Mean V')
    plt.savefig('figures/gridworld/Piteration_meanV_gridworld.png')
if experiment == False:
    policy_iteration(gamma,threshold)
#########################################



################# SARSA #################
def sarsa(gamma,eps,n_episodes,alpha,experiment):
    # initialization
    Q = (1/env.num_states)*np.ones((env.num_states,env.num_actions))
    episodes = []
    pi = np.zeros(env.num_states)
    Gs = []

    for i in range(0,n_episodes+1,1): # new trajectory/episode
        if n_episodes>1000 and np.mod(i,round(n_episodes/100)) == 0:    
            print(i)
        if n_episodes<1000:
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
        ## Plotting the agent's trajectories
        s = env.reset() # choose an initial state randomly
        # Create log to store data from simulation
        log = {
            't': [0],
            's': [s],
            'a': [],
            'r': [],
        }
        done = False
        while not done:
            a = pi[s]
            (s, r, done) = env.step(a)
            log['t'].append(log['t'][-1] + 1)
            log['s'].append(s)
            log['a'].append(a)
            log['r'].append(r)
        # Plot the agent data and save to png file
        plt.figure(plt.gcf().number+1)
        plt.plot(log['t'], log['s'])
        plt.plot(log['t'][:-1], log['a'])
        plt.plot(log['t'][:-1], log['r'])
        plt.title('State, Action, and Reward Trajectories - SARSA')
        plt.xlabel('time step')
        plt.legend(['s', 'a', 'r'])
        plt.savefig('figures/gridworld/sarsa_traj_gridworld.png')

        # Plot the learning curve
        plt.figure(plt.gcf().number+1)
        plt.plot(episodes,Gs)
        plt.title('SARSA Learning Curve')
        plt.xlabel('number of episodes')
        plt.ylabel('G')
        plt.savefig('figures/gridworld/sarsa_g_gridworld.png')
    return(Q,pi,Gs,episodes,n_episodes)
##############################################



################# Q-LEARNING #################
def Qlearning(gamma,eps,n_episodes,alpha,experiment):
    # initialization
    Q = (1/env.num_states)*np.ones((env.num_states,env.num_actions))
    Q_candidates = np.zeros(env.num_actions)
    episodes = []
    pi = np.zeros(env.num_states)
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
        ## Plotting the agent's trajectories
        s = env.reset() # choose an initial state randomly
        # Create log to store data from simulation
        log = {
            't': [0],
            's': [s],
            'a': [],
            'r': [],
        }
        done = False
        while not done:
            a = pi[s]
            (s, r, done) = env.step(a)
            log['t'].append(log['t'][-1] + 1)
            log['s'].append(s)
            log['a'].append(a)
            log['r'].append(r)
        # Plot the agent data and save to png file
        plt.figure(plt.gcf().number+1)
        plt.plot(log['t'], log['s'])
        plt.plot(log['t'][:-1], log['a'])
        plt.plot(log['t'][:-1], log['r'])
        plt.title('State, Action, and Reward Trajectories - Q-Learning')
        plt.xlabel('time step')
        plt.legend(['s', 'a', 'r'])
        plt.savefig('figures/gridworld/qlearn_traj_gridworld.png')
        
        # Plot the learning curve
        plt.figure(plt.gcf().number+1)
        plt.plot(episodes,Gs)
        plt.title('Q-Learning Learning Curve')
        plt.xlabel('number of episodes')
        plt.ylabel('G')
        plt.savefig('figures/gridworld/qlearn_g_gridworld.png')
    return(Q,pi,Gs,episodes,n_episodes)
########################################################



################### TD(0) ESTIMATION ###################
def  TD0(gamma,eps,n_episodes,alpha,experiment):
        # initialization
        V = (1/env.num_states)*np.ones(env.num_states)
        V_q = (1/env.num_states)*np.ones(env.num_states)
        
        # Estimatting SARSA's Value Function
        Q,pi,Gs,episodes,n_episodes = sarsa(gamma,eps,n_episodes,alpha,experiment)
            
        alpha = 1/n_episodes

        for i in range(0,n_episodes+1,1):
            s = env.reset()
            done = False
            while not done:
                a = pi[s]
                (s1, r, done) = env.step(a)
                V[s] = V[s] + alpha*(r + gamma*V[s1] - V[s])
                s = s1
        
        # Estimatting Q-learning's Value Function
        Q_q,pi_q,Gs_q,episodes,n_episodes = Qlearning(gamma,eps,n_episodes,alpha,experiment)
            
        alpha = 1/n_episodes

        for i in range(0,n_episodes+1,1):
            s = env.reset()
            done = False
            while not done:
                a = pi_q[s]
                (s1, r, done) = env.step(a)
                V_q[s] = V_q[s] + alpha*(r + gamma*V_q[s1] - V_q[s])
                s = s1
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
    plt.savefig('figures/gridworld/exp_alpha_sarsa_g_gridworld.png')
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
    plt.savefig('figures/gridworld/exp_alpha_qlearn_g_gridworld.png')
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
    plt.savefig('figures/gridworld/exp_eps_sarsa_g_gridworld.png')
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
    plt.savefig('figures/gridworld/exp_eps_qlearn_g_gridworld.png')
if experiment == True:
    plot_epsilons_qlearn(gamma,eps,n_episodes,alpha,experiment,min_eps,max_eps,n_eps)
#####################################################