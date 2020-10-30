import numpy as np
import psutil
from scipy.signal import convolve, gaussian
import torch
from torch import nn
import os


def get_cum_discounted_rewards(rewards, gamma):
    """
    evaluates cumulative discounted rewards:
    r_t + gamma * r_{t+1} + gamma^2 * r_{t_2} + ...
    """
    cum_rewards = []
    cum_rewards.append(rewards[-1])
    for r in reversed(rewards[:-1]):
        cum_rewards.insert(0, r + gamma * cum_rewards[0])
    return cum_rewards


def play_and_log_episode(env, agent, gamma=0.99, t_max=10000):
    """
    always greedy
    """
    states = []
    v_mc = []
    v_agent = []
    q_spreads = []
    td_errors = []
    rewards = []

    s = env.reset()
    for step in range(t_max):
        states.append(s)
        qvalues = agent.get_qvalues([s])
        max_q_value, min_q_value = np.max(qvalues), np.min(qvalues)
        v_agent.append(max_q_value)
        q_spreads.append(max_q_value - min_q_value)
        if step > 0:
            td_errors.append(
                np.abs(rewards[-1] + gamma * v_agent[-1] - v_agent[-2]))

        action = qvalues.argmax(axis=-1)[0]

        s, r, done, _ = env.step(action)
        rewards.append(r)
        if done:
            break
    td_errors.append(np.abs(rewards[-1] + gamma * v_agent[-1] - v_agent[-2]))

    v_mc = get_cum_discounted_rewards(rewards, gamma)

    return_pack = {
        'states': np.array(states),
        'v_mc': np.array(v_mc),
        'v_agent': np.array(v_agent),
        'q_spreads': np.array(q_spreads),
        'td_errors': np.array(td_errors),
        'rewards': np.array(rewards),
        'episode_finished': np.array(done)
    }

    return return_pack


def play_and_record(initial_state, agent, env, exp_replay, n_steps=1):
    """
    Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer. 
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has done=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time and the state in which the env stays
    """
    s = initial_state
    sum_rewards = 0

    # Play the game for n_steps as per instructions above
    for step in range(n_steps):
        qvalues = agent.get_qvalues([s])
        action = agent.sample_actions(qvalues)[0]
        next_s, r, done, _ = env.step(action)
        sum_rewards += r
        exp_replay.add(s, action, r, next_s, done)
        if done is True:
            s = env.reset()
        else:
            s = next_s

    return sum_rewards, s


def play_and_record_PER(initial_state, agent, env, PER, n_steps=1):
    """
    Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer. 
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has done=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time and the state in which the env stays
    """
    s = initial_state
    sum_rewards = 0

    # Play the game for n_steps as per instructions above
    for step in range(n_steps):
        qvalues = agent.get_qvalues([s])
        action = agent.sample_actions(qvalues)[0]
        next_s, r, done, _ = env.step(action)
        sum_rewards += r
        PER.add(s, action, r, next_s, done, p=PER.sampler.max_priority)
        if done is True:
            s = env.reset()
        else:
            s = next_s

    return sum_rewards, s


def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for _ in range(t_max):
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, _ = env.step(action)
            reward += r
            if done:
                break

        rewards.append(reward)
    return np.mean(rewards)


def img_by_obs(obs, state_dim):
    """
    Unwraps obs by channels.
    observation is of shape [c, h=w, w=h]
    """
    obs_list = [single_obs for single_obs in obs]
    image = np.zeros((state_dim[1], state_dim[0] * state_dim[2]))
    for i, single_obs in enumerate(obs_list):
        image[:, i * state_dim[2]:(i + 1) * state_dim[2]] = single_obs
    return image


def is_enough_ram(min_available_gb=0.1):
    mem = psutil.virtual_memory()
    return mem.available >= min_available_gb * (1024 ** 3)


def linear_decay(init_val, final_val, cur_step, total_steps):
    if cur_step >= total_steps:
        return final_val
    return (init_val * (total_steps - cur_step) +
            final_val * cur_step) / total_steps


def smoothen(values):
    kernel = gaussian(100, std=100)
    # kernel = np.concatenate([np.arange(100), np.arange(99, -1, -1)])
    kernel = kernel / np.sum(kernel)
    return convolve(values, kernel, 'valid')

def conv2d_size_out(size, kernel_size, stride):
    """
    common use case:
    cur_layer_img_w = conv2d_size_out(cur_layer_img_w, kernel_size, stride)
    cur_layer_img_h = conv2d_size_out(cur_layer_img_h, kernel_size, stride)
    to understand the shape for dense layer's input
    """
    return (size - (kernel_size - 1) - 1) // stride  + 1
