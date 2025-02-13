import gymnasium
import gym_gridworlds
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from itertools import permutations
import os
import sys
import importlib
from multiprocessing import Pool
import numpy as np


# working_dir = "/media/adebayo/Windows/UofACanada/study/codes/CMPUT655_RL/Project/project_nstep"
# os.chdir(working_dir)

np.set_printoptions(precision=3, suppress=True)

# https://en.wikipedia.org/wiki/Pairing_function
def cantor_pairing(x, y):
    return int(0.5 * (x + y) * (x + y + 1) + y)

def rbf_features(x: np.array, c: np.array, s: np.array) -> np.array:
    return np.exp(-(((x[:, None] - c[None]) / s[None])**2).sum(-1) / 2.0)

def aggregation_features(state: np.array, centers: np.array) -> np.array:
    distance = ((state[:, None, :] - centers[None, :, :])**2).sum(-1)
    return (distance == distance.min(-1, keepdims=True)) * 1.0  # make it float

def expected_return(env, weights, gamma, episodes=100):
    G = np.zeros(episodes)
    for e in range(episodes):
        s, _ = env.reset(seed=e)
        done = False
        t = 0
        while not done:
            phi = get_phi(s)
            #a = np.dot(phi, weights)
            #a_clip = np.clip(a, env.action_space.low, env.action_space.high)  # this is for the Pendulum
            a = eps_greedy_action(phi, weights, 0)  # this is for the Gridworld
            s_next, r, terminated, truncated, _ = env.step(a)  # replace with a for Gridworld
            done = terminated or truncated
            G[e] += gamma**t * r
            s = s_next
            t += 1
    return G.mean()


def collect_data(env, weights, sigma, n_episodes, eps, seed):
    data = dict()
    data["phi"] = []
    data["a"] = []
    data["r"] = []
    data["s"] = []
    data["s_next"] = []
    data["done"] = []
    for ep in range(n_episodes):
        episode_seed = cantor_pairing(ep, seed)
        s, _ = env.reset(seed=episode_seed)
        done = False
        steps = 0
        while not done:
            phi = get_phi(s)
            # print(phi)
            # exit()
            # a = gaussian_action(phi, weights, sigma)
            a = softmax_action(phi, weights, eps)

            made_noisy_action = False
            if add_action_noise and np.random.random() < rand_act_prob:
                noisy_action = env.action_space.sample()
                made_noisy_action = True

            #a = eps_greedy_action(phi, weights, eps)
            #a_clip = np.clip(a, env.action_space.low, env.action_space.high)  # only for Gaussian policy
            if made_noisy_action:
                s_next, r, terminated, truncated, _ = env.step(noisy_action)
            else:
                s_next, r, terminated, truncated, _ = env.step(a)

            if add_reward_noise:
                if reward_noise_sd > 0.0:
                    # print('reward')
                    # print('before', r)
                    r += np.random.normal() * reward_noise_sd
                    # print('After', r)
            if add_observation_noise:
                # print('before', s_next)
                if obs_noise > 0.0:
                    if np.random.random() < obs_noise:
                        # print('obs noise')
                        s_next = np.array([np.random.uniform(state_low[i], state_high[i]) for i in range(len(state_low))])
            
            done = terminated or truncated
            data["phi"].append(phi)
            data["a"].append(a)
            data["r"].append(r)
            data["s"].append(s)
            data["s_next"].append(s_next)
            data["done"].append(terminated or truncated)
            s = s_next

            steps += 1

    return data

def get_policy(weights):
    pi = []
    for row in range(n_rows):
        new_row = []
        for col in range(n_cols):
            phi = get_phi(np.array([row, col]))
            new_row.append(eps_greedy_action(phi, weights, 0))
        pi.append(new_row)
    return np.array(pi)


def eps_greedy_action(phi, weights, eps):
    if np.random.rand() < eps:
        return np.random.randint(n_actions)
    else:
        Q = np.dot(phi, weights).ravel()
        best = np.argwhere(Q == Q.max())
        i = np.random.choice(range(best.shape[0]))
        return best[i][0]

def softmax_probs(phi, weights, eps):
    q = np.dot(phi, weights)
    # this is a trick to make it more stable
    # see https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    q_exp = np.exp((q - np.max(q, -1, keepdims=True)) / max(eps, 1e-12))
    probs = q_exp / q_exp.sum(-1, keepdims=True)
    return probs

def softmax_action(phi, weights, eps):
    probs = softmax_probs(phi, weights, eps)
    return np.random.choice(weights.shape[1], p=probs.ravel())

def dlog_softmax_probs_full(phi, weights, eps, act):
    # implement log-derivative of pi
    n_samples, n_features = phi.shape
    # print(n_samples, n_features)

    probs = softmax_probs(phi, weights, eps)
    # print('PROBS', probs.shape)
   
    phi_sa = phi[..., None].repeat(n_actions, -1)
    # print('PHI_SA', phi_sa.shape)

    mask = np.zeros((n_samples, n_features, n_actions))

    for i in range(n_samples):
        action = act[i][0]
        mask[i, :, action] = 1.0  
        # mask[:, act] = 1.0



    dlog = phi_sa * mask - phi_sa*(probs[:,None,:])
    # print('DLOG', dlog.shape)

    return dlog

def dlog_softmax_probs(phi, weights, eps, act):
    # implement log-derivative of pi
    n_samples, n_features = phi.shape
    # print(n_samples, n_features)

    probs = softmax_probs(phi, weights, eps)
    # print('PROBS', probs.shape)
   
    phi_sa = phi[..., None].repeat(n_actions, -1)[0]
    # print('PHI_SA', phi_sa.shape)

    mask = np.zeros((n_features, n_actions))

    for i in range(n_samples):
        # action = act[i][0]
        # mask[i, :, action] = 1.0  
        mask[:, act] = 1.0



    dlog = phi_sa * mask - phi_sa*(probs)
    # print('DLOG', dlog.shape)

    return dlog

def gaussian_action(phi, weights, sigma: np.array):
    mu = np.dot(phi, weights)
    return np.random.normal(mu, sigma**2)

def dlog_gaussian_probs(phi, weights, sigma, action: np.array):
    # implement log-derivative of pi with respect to the mean only
    mu = np.dot(phi, weights)
    return (action - mu) / (sigma ** 2) * phi



def reinforce(n = 5, MC_update = 0, seed=1):
    weights_p = np.zeros((phi_dummy.shape[1], n_actions))
    weights_v = np.zeros((phi_dummy.shape[1], n_actions)) #DIFFERENT DIMENSION THAN WEIGHTS_P????
    sigma = 2.0  # for Gaussian
    eps = 1.0  # softmax temperature, DO NOT DECAY
    tot_steps = 0
    episode_count = 0 
    exp_return_history = np.zeros(max_steps)
    weight_variance_history = np.zeros(max_steps)
    theta_variance_history = np.zeros(max_steps)
    performance_variance_history = np.zeros(max_steps)
    delta_error_history = np.zeros(max_steps)
    reward_history = np.zeros(max_steps)
    episode_history = np.zeros(max_steps)
    average_return_history = np.zeros(max_steps)
    

    exp_return = expected_return(env_eval, weights_p, gamma, episodes_eval)

    pbar = tqdm(total=max_steps)
    data = dict()

    metrics = {
        "reward": reward_history,
        "average_return": average_return_history,
        "expectation": exp_return_history,
        "weights_stability": weight_variance_history,
        "theta_stability": theta_variance_history,
        "performance": performance_variance_history,
        "delta_error": delta_error_history,
        "episode": episode_history,
    }

    
    returns = np.zeros(max_steps)
    deltas = np.zeros(max_steps)

    while tot_steps < max_steps:
        # collect data from a single episode
        
        data = collect_data(env, weights_p, sigma, 1, eps, seed=seed)
        phi = np.vstack(data['phi'])
        r = np.vstack(data['r'])
        # r = np.array(data['r'])
        a = np.vstack(data['a'])
        s = np.vstack(data["s"])
        s_next = np.vstack(data["s_next"])
        done = np.vstack(data['done'])

        ep_steps = len(r) # steps taken while collecting data
        episode_count += 1  # Increment episode count
        #print(f"Steps in episodes {ep_steps}")
        # Ensure arrays do not exceed max_steps
        if tot_steps + ep_steps > max_steps:
            ep_steps = max_steps - tot_steps 

        # Performance variance calculation
        average_return = np.mean(returns)
        performance_variance = np.var(returns)
        theta_variance = np.var(weights_p)
        weight_variance = np.var(weights_v)
        
        

        delta_error_history[tot_steps: tot_steps + ep_steps] = deltas[:ep_steps]
        theta_variance_history[tot_steps: tot_steps + ep_steps] = theta_variance
        weight_variance_history[tot_steps: tot_steps + ep_steps] = weight_variance
        performance_variance_history[tot_steps: tot_steps + ep_steps] = performance_variance
        reward_history[tot_steps: tot_steps + ep_steps] = returns[:ep_steps]
        average_return_history[tot_steps: tot_steps + ep_steps] = average_return
        episode_history[tot_steps: tot_steps + ep_steps] =   ep_steps    
        
        
        # returns = np.zeros(max_steps)
        # deltas = np.zeros(max_steps)
        # if n==0, do MC
        if n == 0:
            # compute full sample return from each state visited in episode
            G = np.zeros((ep_steps))
            
            G[-1] = r[-1]
            for t in range(ep_steps - 2, -1, -1):
                G[t] = r[t] + gamma*G[t+1]
            
            
                
            # once we have computed return, compute delta and update weights
            for t in range(len(G)):

                if MC_update == 0:
                    fact = (1/(len(G)-t))

                else:
                    fact = (1/len(G))

                delta = G[t] - np.dot(get_phi(s[t]), weights_v).max(-1)
                weights_v[:, a[t]] += (alpha_v*phi[t]*delta).reshape(weights_v.shape[0], 1)*fact # (1/len(G))#*(1/(len(G)-t))#*(1/len(G))#*(1/(len(G)-t)) #*(1/len(G))
                weights_p += alpha_p*(gamma**t)*delta*dlog_softmax_probs(get_phi(s[t]), weights_p, eps, a[t])*fact # (1/len(G))#*(1/(len(G)-t))#*(1/len(G)) #*(1/(len(G)-t)) #*(1/len(G))  
                deltas[tot_steps + t] = delta
                returns[tot_steps + t] = G[t]
                #print(delta)
        
                
        # else, do n-step
        else:
            
            # for each sample of episode
            for t in range(ep_steps):

                # compute n-step rollout
                G = r[t]
                power = 1
                for k in range(t+1, t + n):
                    if k < ep_steps:
                        G += (gamma**power)*r[k]
                        power += 1
                    else:
                        break
                # compute delta
                # if we have not reached the end of the episode from n-step rollout
                if t + power < ep_steps:
                    
                    delta = G + (gamma**power)*np.dot(get_phi(s[t + power]), weights_v).max(-1) - np.dot(get_phi(s[t]), weights_v).max(-1)
                    #print(f"returns: {G + (gamma**power)*np.dot(get_phi(s[t + power]), weights_v).max(-1)}")
                    returns[tot_steps + t + power] = G + (gamma**power) * np.dot(get_phi(s[t + power]), weights_v).max(-1).item()
                    deltas[tot_steps + t +power] = delta
                # if we have reached the end of the episode from n-step rollout
                else:
                    delta = G - np.dot(get_phi(s[t]), weights_v).max(-1)
                    deltas[tot_steps + t] = delta
                    returns[tot_steps + t] = G
                # update weights
                weights_v[:, a[t]] += (alpha_v*get_phi(s[t])*delta).reshape(weights_v.shape[0], 1)*(1/power) 
                weights_p += alpha_p*(gamma**t)*delta*dlog_softmax_probs(get_phi(s[t]), weights_p, eps, a[t])*(1/power)


        

        sigma = max(sigma - ep_steps / max_steps, 0.1)

    
        exp_return_history[tot_steps : tot_steps + ep_steps] = exp_return

        tot_steps += ep_steps

        exp_return = expected_return(env_eval, weights_p, gamma, episodes_eval)

        pbar.set_description(
                f"G: {exp_return:.3f}"
            )
        pbar.update(ep_steps)


    pbar.close()
    

    if n == 0:
        print('Results for MC')
    else:
        print('Results for n-step with n =', n)

    print('tot_steps:', tot_steps)
    if n_rows is not None:
        print(get_policy(weights_p))



    # Store the metrics
    metrics = {
        "reward": reward_history,
        "average_return": average_return_history,
        "expectation": exp_return_history,
        "weights_stability": weight_variance_history,
        "theta_stability": theta_variance_history,
        "performance": performance_variance_history,
        "delta_error": delta_error_history,
        "episode": episode_history,
    }

    return metrics

# https://stackoverflow.com/a/63458548/754136
def smooth(arr, span):
    re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")
    re[0] = arr[0]
    for i in range(1, span + 1):
        re[i] = np.average(arr[: i + span])
        re[-i] = np.average(arr[-i - span :])
    return re

def error_shade_plot(ax, data, stepsize, smoothing_window=1, **kwargs):
    y = np.nanmean(data, 0)
    x = np.arange(len(y))
    x = [stepsize * step for step in range(len(y))]
    if smoothing_window > 1:
        y = smooth(y, smoothing_window)
    (line,) = ax.plot(x, y, **kwargs)
    error = np.nanstd(data, axis=0)
    if smoothing_window > 1:
        error = smooth(error, smoothing_window)
    error = 1.96 * error / np.sqrt(data.shape[0])
    ax.fill_between(x, y - error, y + error, alpha=0.2, linewidth=0.0, color=line.get_color())




#env_ids = ["Gym-Gridworlds/DangerMaze-6x6-v0"]
#env_ids = ["Gym-Gridworlds/CliffWalk-4x12-v0", "Gym-Gridworlds/CliffWalk-5x5-v0", "Gym-Gridworlds/DangerMaze-6x6-v0", "Gym-Gridworlds/Penalty-3x3-v0"]
#env_ids = ["MountainCar-v0"]
#env_ids = ["CartPole-v1", "MountainCar-v0"]
add_reward_noise = True
add_observation_noise = True
add_action_noise = True
# noises = [0.01, 0.05, 0.1]
# noise_combos = [(0.0, 0.0, 0.0)]
# for noise in noises:
#     noise_combos.append((noise, 0.0, 0.0))
#     noise_combos.append((0.0, noise, 0.0))
#     noise_combos.append((0.0, 0.0, noise))


# noise_combos = [(0.0, 0.1, 0.0), (0.0, 0.5, 0.0)]


# noise_combos = [(0.0, 0.0, 0.0)]

# print(noise_combos)
# exit()


import multiprocessing as mp
import numpy as np
import gymnasium

def process_seed(seed, env, n, alpha, MC_update):
    """
    Process a single seed to collect metrics using the REINFORCE algorithm.
    """
    np.random.seed(seed)  # Set the random seed for reproducibility
    
    # Your existing logic here (e.g., reinforcement learning steps)
    metrics = reinforce(n, MC_update, seed=seed)  # Call your reinforcement function
    
    return {
        "seed": seed,
        "expectation": metrics["expectation"],
        "reward": metrics["reward"],
        "average_return": metrics["average_return"],
        "weights_stability": metrics["weights_stability"],
        "theta_stability": metrics["theta_stability"],
        "performance": metrics["performance"],
        "delta_error": metrics["delta_error"],
        "episode": metrics["episode"],
    }


# Main script
if __name__ == "__main__":
    n_processes = 24  # Number of processes to use
    n_seeds = 50
    seeds = list(range(n_seeds))  

    # alphas = [0.1, 0.01]
    alphas = [0.01]
    # updates = [0, 1]
    updates = [0]

    #env_ids = ["CartPole-v1", "MountainCar-v0"]
    env_ids = ["MountainCar-v0"]

    noise_combos = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1),
                (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.05), 
                (0.75, 0.0, 0.0), (0.0, 0.75, 0.0), (0.0, 0.0, 0.01)]
    noise_combos = [(10.0, 0.0, 0.0)]

    noise_combos = [(1.0, 0.0, 0.0), (0.0, 0.25, 0.0)]
    for env_id in env_ids:

        for alpha in alphas:
            for MC_update in updates: 
                for reward_noise_sd, rand_act_prob, obs_noise in noise_combos:
                    
                    if env_id in ["CartPole-v1","MountainCar-v0"]:
                        env = gymnasium.make(env_id, max_episode_steps= 500)
                        env_eval = gymnasium.make(env_id, max_episode_steps=500)
                    else:
                        env = gymnasium.make(env_id, coordinate_observation=True,distance_reward=True, max_episode_steps=1000, reward_noise_std = reward_noise_sd, random_action_prob = rand_act_prob, observation_noise = obs_noise)
                        env_eval = gymnasium.make(env_id, coordinate_observation=True, max_episode_steps=30)  # 10 steps only for faster eval
                    episodes_eval = 1 # max expected return will be 0.941
                    state_dim = env.observation_space.shape[0]
                    n_actions = env.action_space.n

                    # automatically set centers and sigmas
                    if env_id in ["CartPole-v1"]:
                        n_centers = [5] * state_dim
                    elif env_id in ["MountainCar-v0"]:
                        n_centers = [10] * state_dim
                    else:
                        n_centers = [20] * state_dim
                    #n_centers = [20, 20]
                    print(n_centers)
                    if env_id == 'CartPole-v1':
                        state_low = np.array([-4.8, -3, -0.419, -3])
                        state_high = np.array([4.8, 3, 0.419, 3])
                    else:
                        state_low = env.observation_space.low
                        state_high = env.observation_space.high

                    centers = np.array(
                        np.meshgrid(*[
                            np.linspace(
                                state_low[i] - (state_high[i] - state_low[i]) / n_centers[i] * 0.1,
                                state_high[i] + (state_high[i] - state_low[i]) / n_centers[i] * 0.1,
                                n_centers[i],
                            )
                            for i in range(state_dim)
                        ])
                    ).reshape(state_dim, -1).T
                    sigmas = (state_high - state_low) / np.asarray(n_centers) * 0.75 + 1e-8  # change sigmas for more/less generalization
                    if env_id in ["CartPole-v1","MountainCar-v0"]:
                        get_phi = lambda state : rbf_features(state.reshape(-1, state_dim), centers, sigmas)  # reshape because feature functions expect shape (N, S)
                    else:
                        get_phi = lambda state : aggregation_features(state.reshape(-1, state_dim), centers)
                    phi_dummy = get_phi(env.reset()[0])  # to get the number of features

                    # hyperparameters
                    gamma = 0.99
                    gamma_eval = 1.0
                    alpha_p = alpha
                    alpha_v = alpha

                    if env_id == "Gym-Gridworlds/Penalty-3x3-v0":
                        world = "Penalty-3x3-v0"
                        max_steps = 1000000
                        n_rows = 3
                        n_cols = 3

                    elif env_id == "Gym-Gridworlds/CliffWalk-4x12-v0":
                        world = "CliffWalk-4x12-v0"
                        max_steps = 1000000
                        n_rows = 4
                        n_cols = 12

                    elif env_id == "Gym-Gridworlds/CliffWalk-5x5-v0":
                        world = "CliffWalk-5x5-v0"
                        max_steps = 1000000
                        n_rows = 5
                        n_cols = 5

                    elif env_id == "Gym-Gridworlds/DangerMaze-6x6-v0":
                        world = "DangerMaze-6x6-v0"
                        max_steps = 1000000
                        n_rows = 5
                        n_cols = 6

                    elif env_id == "CartPole-v1":
                        world = "CartPole-v1"
                        max_steps = 600000
                        n_rows = None

                    elif env_id == "MountainCar-v0":
                        world = "MountainCar-v0"
                        max_steps = 600000
                        n_rows = None

                    
                    n_vals = [0, 1, 2, 3, 10]
                

                    results_exp_ret = np.zeros((
                        len(n_vals),
                        n_seeds,
                        max_steps,
                    ))
                    reward = np.zeros((
                        len(n_vals),
                        n_seeds,
                        max_steps,
                    ))
                    average_return = np.zeros((
                        len(n_vals),
                        n_seeds,
                        max_steps,
                    ))
                    weights_stability = np.zeros((
                        len(n_vals),
                        n_seeds,
                        max_steps,
                    ))
                    theta_stability  = np.zeros((
                        len(n_vals),
                        n_seeds,
                        max_steps,
                    ))
                    performance = np.zeros((
                        len(n_vals),
                        n_seeds,
                        max_steps,
                    ))
                    delta_error = np.zeros((
                        len(n_vals),
                        n_seeds,
                        max_steps,
                    ))
                    episode = np.zeros((
                        len(n_vals),
                        n_seeds,
                        max_steps,
                    ))

                    index = 0

                    # Start multiprocessing
                    for n in n_vals:
                        with Pool(n_processes) as pool:
                            # Distribute seeds across processes
                            results = pool.starmap(process_seed, [(seed, env, n, alpha, MC_update) for seed in seeds])


                        # Collect results
                        for result in results:
                            seed = result["seed"]
                            results_exp_ret[index, seed] = result["expectation"]
                            reward[index, seed] = result["reward"]
                            average_return[index, seed] = result["average_return"]
                            weights_stability[index, seed] = result["weights_stability"]
                            theta_stability[index, seed] = result["theta_stability"]
                            performance[index, seed] = result["performance"]
                            delta_error[index, seed] = result["delta_error"]
                            episode[index, seed] = result["episode"]
                            print(f"Processed seed: {seed}")

                        index += 1
                    

                    # Save the results and metadata
                    save_filename = f"{world}_{n_seeds}_{alpha}_{reward_noise_sd}_{rand_act_prob}_{obs_noise}_results.npz"
                    np.savez(
                        save_filename,
                        results_exp_ret=results_exp_ret,
                        reward = reward,
                        average_return = average_return,
                        weights_stability = weights_stability,
                        theta_stability = theta_stability,
                        performance = performance,
                        delta_error = delta_error,
                        episode = episode,
                    )
                    print(f"Results saved to {save_filename}")
           
