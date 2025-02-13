import gymnasium
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from IPython.display import clear_output

np.set_printoptions(precision=3)

# Initialize environment
env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")
n_states = env.observation_space.n
n_actions = env.action_space.n

# Initialize rewards, transitions, and terminal states
R = np.zeros((n_states, n_actions))
P = np.zeros((n_states, n_actions, n_states))
T = np.zeros((n_states, n_actions))

env.reset()
for s in range(n_states):
    for a in range(n_actions):
        env.unwrapped.set_state(s)
        s_next, r, terminated, _, _ = env.step(a)
        R[s, a] = r
        P[s, a, s_next] = 1.0
        T[s, a] = terminated

P = P * (1.0 - T[..., None])  # No transitions after terminal states

# Bellman Q-value computation
def bellman_q(pi, gamma, max_iter=1000):
    delta = np.inf
    iter = 0
    Q = np.zeros((n_states, n_actions))
    be = np.zeros((max_iter))
    while delta > 1e-5 and iter < max_iter:
        Q_new = R + (np.dot(P, gamma * (Q * pi)).sum(-1))
        delta = np.abs(Q_new - Q).sum()
        be[iter] = delta
        Q = Q_new
        iter += 1
    return Q

# Generate an episode following the current policy
def episode(env, Q, eps, seed=None):
    data = {"s": [], "a": [], "r": []}
    s, _ = env.reset(seed=seed)
    done = False
    while not done:
        a = eps_greedy_action(Q, s, eps)
        s_next, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        data["s"].append(s)
        data["a"].append(a)
        data["r"].append(r)
        s = s_next
    return data

# Epsilon-greedy policy
def eps_greedy_probs(Q, eps):
    nA = len(Q)
    probs = np.ones(nA) * eps / nA
    best_action = np.argmax(Q)
    probs[best_action] += (1.0 - eps)
    return probs

def eps_greedy_action(Q, s, eps):
    probs = eps_greedy_probs(Q[s], eps)
    action = np.random.choice(np.arange(len(probs)), p=probs)
    return action



# Monte Carlo method for policy evaluation and control
def monte_carlo(env, Q, gamma, decay_factor, max_steps, episodes_per_iteration, use_is=False):
    eps = 1.0  # Start with epsilon = 1
    #bellman_errors = np.zeros(max_steps)
    pi = np.ones((n_states, n_actions)) / n_actions  # Initialize policy uniformly
    total_steps = 0
    returns = np.empty((n_states, n_actions), dtype=object)
    for i in range(n_states):
        for j in range(n_actions):
            returns[i,j] =[]
    bellman_errors = []
    bellman_error = np.abs(Q - bellman_q(pi, gamma)).sum()
    
    while total_steps < max_steps:
        for ep in range(episodes_per_iteration):
            data = episode(env, Q, eps, seed=None)
            episdoe_length = len(data['s'])
            total_steps += episdoe_length
            G = 0
            weights = 1
            target_eps = 0.01

            if use_is:
                
                target_policy = np.array([eps_greedy_probs(Q[s], target_eps) for s in range(n_states)])
                C = {(s, a): 0 for s in range(n_states) for a in range(n_actions)}
                for t in range(len(data["s"])- 1, -1, -1):
                    s = data["s"][t]
                    a = data["a"][t]
                    G = data["r"][t] + gamma * G
                    C[s,a] = C[s,a] + weights
                    returns[(s, a)].append(G) 
                    prob_behavior = eps_greedy_probs(Q[s], eps)[a]
                    prob_target = target_policy[s, a]
                    weights=prob_target / prob_behavior
                    Q[s,a]  = Q[s,a]  + weights/C[s,a] * (G - Q[s,a])
                
                    weights = np.array(weights)
                
                    Q[s, a] = np.mean(returns[(s, a)])
                    bellman_errors.append(bellman_error)
                    
        
                    
            else:
                for t in range(len(data["s"]) - 1, -1, -1):
                    G = data["r"][t] + gamma * G
                    s = data["s"][t]
                    a = data["a"][t]
                    returns[(s, a)].append(G)                    

                    # Q[s, a] = np.mean(returns[s, a])
                    Q[s, a] = np.mean(returns[(s, a)])
                    bellman_errors.append(bellman_error)

                

            

            # Update policy using current Q-values
            eps = max(eps - decay_factor/max_steps * len(data["s"]), 0.01)

            pi = np.array([eps_greedy_probs(Q[s], eps) for s in range(n_states)])
            
       
            # Compute Bellman error
            bellman_error = np.abs(Q - bellman_q(pi, gamma)).sum()
            bellman_errors.append(bellman_error)
            #print(bellman_error)
         
    return Q, bellman_errors


init_value = 0.0
gamma = 0.9
max_steps = 2000
horizon = 10
episodes_per_iteration = 10
decay_factor = 1
seeds = np.arange(10)

# Q = np.zeros((n_states, n_actions))
# Q, bellman_error = monte_carlo(env, Q, gamma, decay_factor, max_steps, episodes_per_iteration, use_is=False)
# plt.plot(bellman_error)
# plt.show()
# exit()
# Plotting function
def error_shade_plot(ax, data, stepsize, **kwargs):
    y = np.nanmean(data, 0)
    x = np.arange(len(y)) * stepsize
    line, = ax.plot(x, y, **kwargs)
    error = np.nanstd(data, axis=0)
    error = 1.96 * error / np.sqrt(data.shape[0])
    ax.fill_between(x, y - error, y + error, alpha=0.2, linewidth=0.0, color=line.get_color())

# Simulation parameters
init_value = 0.0
gamma = 0.9
max_steps = 2000
horizon = 10
episodes_per_iteration = [1, 10, 50]
decays = [1, 2, 5]
seeds = np.arange(50)

results = np.zeros((len(episodes_per_iteration), len(decays), len(seeds), max_steps))

fig, axs = plt.subplots(1, 2)
plt.ion()
plt.show()

use_is = True
for ax, reward_noise_std in zip(axs, [0.0, 3.0]):
    ax.set_prop_cycle(
        color=["red", "green", "blue", "black", "orange", "cyan", "brown", "gray", "pink"]
    )
    ax.set_xlabel("Steps")
    ax.set_ylabel("Absolute Bellman Error")
    env = gymnasium.make(
        "Gym-Gridworlds/Penalty-3x3-v0",
        max_episode_steps=horizon,
        reward_noise_std=reward_noise_std,
    )
    for j, episodes in enumerate(episodes_per_iteration):
        for k, decay in enumerate(decays):
            for seed in seeds:
                np.random.seed(seed)
                Q = np.zeros((n_states, n_actions)) + init_value
                Q, be = monte_carlo(env, Q, gamma, decay, max_steps, episodes, use_is)
                be = be[:max_steps]
                results[j, k, seed] = be
            error_shade_plot(
                ax,
                results[j, k],
                stepsize=1,
                label=f"Episodes: {episodes}, Decay: {decay}",
            )
            #print(Q)
            ax.legend()
            plt.draw()
            plt.pause(0.001)

plt.ioff()
plt.show()
