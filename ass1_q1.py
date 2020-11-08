import gym
import numpy as np
from tqdm import tqdm
env = gym.make("FrozenLake-v0")
# Build and initialize lookup table
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Initialize parameters
params = {'alpha': 0.1,
          'discount_factor': .0618,
          'epsilon': 0.4,
          'min_epsilon': 0.05,
          'decay': 0.8,
          'train_episodes': 5000,
          'max_steps': 100}

actions_dict = {0: 'Left',
                1: 'Down',
                2: 'Right',
                3: 'Up'}


def q_learning(env, Q, params, actions_dict, verbose=False):
    """

    :param env:
    :param Q:
    :param params:
    :param actions_dict:
    :return:
    """
    r_per_episode = []
    step_per_episode = []
    for episode in tqdm(range(params['train_episodes'])):
        current_reward = 0
        state = env.reset()
        for step in range(params['max_steps']):
            if verbose:
                env.render()
            rand = np.random.rand()
            if params['epsilon'] > rand:
                action = env.action_space.sample()
                # params['epsilon'] #todo: ask in forum about the update of epsilon
            else:
                action = np.argmax(Q[state, :])

            # get reward
            new_state, reward, terminal, info = env.step(action)
            current_reward += reward

            # updating the q-function using the bellman equation
            Q[state, action] = Q[state, action] + params['alpha'] * (
                    reward + params['discount_factor'] * np.max(Q[new_state, :] - Q[state, action])
            )

            state = new_state

            if terminal:
                break
        r_per_episode.append(current_reward)
        step_per_episode.append(step)
        params['epsilon'] = max(params['min_epsilon'], params['epsilon'] * params['decay'])

    return r_per_episode, step_per_episode

x,y = q_learning(env=env,Q=Q,actions_dict=actions_dict,params=params,verbose=False)



