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
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop
from tqdm import tqdm

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


# def get_model(n_input=4, n_output=2, n_hidden_layers=3, n_neurons=64, learning_rate=0.001):
#     model = Sequential()
#     model.add(Dense(n_neurons, input_dim=n_input, activation='relu'))
#     for layer_index in range(n_hidden_layers - 1):
#         model.add(Dense(n_neurons, activation='relu'))
#     model.add(Dense(n_output, activation='linear', name='action'))

#     optimizer = Adam(lr=learning_rate)
#     model.compile(loss='mse', optimizer=optimizer)
#     return model


def get_model(input_shape=tuple([4]), action_space=2, learning_rate = 0.00025):
    X_input = Input(input_shape)

    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)

    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    # Output Layer with # of actions: 2 nodes (left, right)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs=X_input, outputs=X, name='CartPole_DQN_model')
    model.compile(loss="mse", optimizer=RMSprop(lr=learning_rate, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    return model


class DQN:
    def __init__(self, alpha: float, discount_factor: float, epsilon: float, min_epsilon: float, decay: float,
                 train_episodes: int, max_steps: int, batch_size: int,
                 replay_buffer_size: int, target_update: int, result_folder: str, train_start: int):
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
        self.target_model = get_model(learning_rate = alpha)
        self.Q = get_model(learning_rate = alpha)
        self.result_folder = result_folder
        self.train_start = train_start

    def act(self, state):
        return np.argmax(self.Q.predict(state)[0])

    def learn_environment(self, env, verbose=False):
        self.update_target_model()
        epsilon = self.epsilon
        acc_rewards = []
        epsilons = []
        episode = -1

        for _ in tqdm(range(self.train_episodes)):
            episode += 1
            if ((episode + 1) % 5) == 0:
                self.save_model()
                self.save_list(acc_rewards, 'rewards')
                self.save_list(epsilons, 'epsilons')
                self.save_list([episode], 'am_alive')

            state = env.reset()
            state = np.reshape(state, (1, 4))

            current_reward = 0
            for step in range(self.max_steps):
                if verbose:
                    env.render()

                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = self.act(state)

                new_state, reward, terminal, info = env.step(action)

                new_state = new_state.reshape(1, 4)
                current_reward += reward

                if not terminal or step == env._max_episode_steps - 1:
                    reward = reward
                else:
                    reward = -100

                self.replay_buffer.push(state, action, new_state, reward, terminal)
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
        if len(self.replay_buffer.memory) < self.train_start:
            return

        samples = self.replay_buffer.sample(self.batch_size)
        states = np.zeros((self.batch_size, 4))
        next_states = np.zeros((self.batch_size, 4))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            states[i] = samples[i][0][0]
            action.append(samples[i][1])
            next_states[i] = samples[i][2][0]
            reward.append(samples[i][3])
            done.append(samples[i][4])

        target = self.Q.predict(states)
        target_next = self.target_model.predict(next_states)

        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (np.amax(target_next[i]))

        self.Q.fit(states, target, batch_size=self.batch_size, verbose=False)
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


res_folder = 'res6'
params = {'alpha': 0.001,
          'discount_factor': 0.95,
          'epsilon': 1,
          'min_epsilon': 0.0005,
          'decay': 0.99,
          'train_episodes': 1000,
          'max_steps': 500,
          'batch_size': 256,
          'replay_buffer_size': 8192,
          'target_update': 10,
          'result_folder': res_folder,
          'train_start': 1024
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
