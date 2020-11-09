from collections import namedtuple
import tensorflow as tf
import os
from gym.envs.registration import EnvRegistry
from keras.models import Sequential
import numpy as np
from keras.layers import Dense
import gym
from keras.optimizers import Adam
import random
import json

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'is_terminal'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def get_model(n_input=4, n_output=2, n_hidden_layers=3, n_neurons=64, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(n_neurons, input_dim=n_input, activation='relu'))
    for layer_index in range(n_hidden_layers - 1):
        model.add(Dense(n_neurons, activation='relu'))
    model.add(Dense(n_output, activation='linear', name='action'))

    optimizer = Adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model


class DQN:
    def __init__(self, alpha: float, discount_factor: float, epsilon: float, min_epsilon: float, decay: float,
                 train_episodes: int, max_steps: int, batch_size: int,
                 replay_buffer_size: int, target_update: int, result_folder: str):
        self.alpha = alpha
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.train_episodes = train_episodes
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.target_update_counter = target_update
        self.replay_buffer: ReplayMemory = ReplayMemory(replay_buffer_size)
        self.replay_counter = 0
        self.target_model = get_model()
        self.Q = get_model()
        self.result_folder = result_folder

    def act(self, state):
        return np.argmax(self.target_model.predict(state)[0])

    def learn_environment(self, env, verbose=False):
        epsilon = self.epsilon

        acc_rewards = []
        epsilons = []

        episode = -1

        while True:
            episode += 1
            current_reward = 0
            if ((episode + 1) % 5) == 0:
                self.save_model()
                self.save_list(acc_rewards, 'rewards')
                self.save_list(epsilons, 'epsilons')
                self.save_list([episode], 'am_alive')

            if episode == 1001:
                break

            state = env.reset().reshape(1, 4)
            for step in range(self.max_steps):
                if verbose:
                    env.render()

                if np.random.rand() < params['epsilon']:
                    action = env.action_space.sample()
                else:
                    action = self.act(state)

                new_state, reward, terminal, info = env.step(action)

                #                 if terminal:
                #                     reward = -100

                new_state = new_state.reshape(1, 4)
                current_reward += reward

                self.replay_buffer.push(state, action, np.zeros(state.shape), reward, terminal)
                self.replay_training()

                if ((self.replay_counter + 1) % self.target_update_counter) == 0:
                    self.update_target_model()

                state = new_state

                if terminal:
                    break

            acc_rewards.append(current_reward)
            epsilon = max(self.min_epsilon, self.decay * epsilon)
            epsilons.append(epsilon)

    def replay_training(self):
        # not enough for a batch
        if self.replay_buffer.position < self.batch_size:
            return

        samples = self.replay_buffer.sample(self.batch_size)

        for state, action, new_state, reward, terminal in samples:
            q_update = reward

            if not terminal:
                q_update = (reward + self.discount_factor * np.amax(self.target_model.predict(new_state)[0]))

            q_values = self.Q.predict(state)
            q_values[0][action] = q_update

            self.Q.fit(state, q_values, verbose=0)

        self.replay_counter += 1

    def update_target_model(self):
        self.target_model.set_weights(self.Q.get_weights())

    def save_model(self):
        self.Q.save(f'{self.result_folder}/Q.h5')
        self.target_model.save(f'{self.result_folder}/target.h5')

    def save_list(self, arr, name):
        with open(f'{self.result_folder}/{name}.txt', 'w+') as f:
            for val in arr:
                f.write(str(val) + '\n')


res_folder = 'res2'
params = {'alpha': 0.001,
          'discount_factor': 0.95,
          'epsilon': 1,
          'min_epsilon': 0.01,
          'decay': 0.99,
          'train_episodes': 150,
          'max_steps': 500,
          'batch_size': 512,
          'replay_buffer_size': 4096,
          'target_update': 5,
          'result_folder': res_folder
          }

try:
    os.mkdir(res_folder)
except:
    pass

with open(f'{res_folder}/expr_params.json', 'w+') as f:
    json.dump(params, f)

dqn = DQN(**params)
env = gym.make('CartPole-v1')
dqn.learn_environment(env)
dqn.save_model()