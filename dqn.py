from collections import namedtuple
import tensorflow as tf
from gym.envs.registration import EnvRegistry
from keras.models import Sequential
import numpy as np
from keras.layers import Dense
import gym
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from tqdm import tqdm
import random

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


def get_model(n_input=4, n_output=2, n_hidden_layers=2, n_neurons=16, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(n_neurons, input_dim=n_input, activation='relu'))
    for layer_index in range(n_hidden_layers - 1):
        model.add(Dense(n_neurons, activation='relu'))
    model.add(Dense(n_output))

    optimizer = Adam(lr=learning_rate)
    model.compile(loss='MSE', optimizer=optimizer)
    return model


class DQN:
    def __init__(self, alpha: float, discount_factor: float, epsilon: float, min_epsilon: float, decay: float,
                 train_episodes: int, max_steps: int, batch_size: int,
                 replay_buffer_size: int, target_update: int):
        self.alpha = alpha
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.train_episodes = train_episodes
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.target_update = target_update
        self.replay_buffer: ReplayMemory = ReplayMemory(replay_buffer_size)

        self.target_model = get_model()
        self.Q = get_model()

    def act(self, state):
        return np.argmax(self.Q.predict(state)[0])

    def learn_environment(self, env, verbose=False):
        epsilon = self.epsilon
        current_reward = 0
        for episode in tqdm(range(self.train_episodes)):
            state = env.reset().reshape(1, 4)
            step = 0

            for step in range(self.max_steps):
                if verbose:
                    env.render()
                rand = np.random.rand()
                if params['epsilon'] > rand:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(self.target_model.predict(state)[0])

                new_state, reward, terminal, info = env.step(action)
                new_state = new_state.reshape(1, 4)
                current_reward += reward

                self.replay_buffer.push(state, action, new_state, reward, terminal)
                self.replay_training()

                state = new_state
                epsilon = max(self.min_epsilon, self.decay * epsilon)

                if terminal:
                    break

    def replay_training(self):

        # not enough for a batch
        if self.replay_buffer.position < self.batch_size:
            return

        samples = self.replay_buffer.sample(self.batch_size)
        for index, sample in enumerate(samples):
            state, action, new_state, reward, terminal = sample
            target = self.target_model.predict(state)

            if terminal:
                target[0][action] = reward
            else:
                debug = self.target_model.predict(new_state)
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.discount_factor

            self.Q.fit(state, target, epochs=1, verbose=0)

            if ((index + 1) % self.target_update) == 0:
                self.target_model.set_weights(self.Q.get_weights())


params = {'alpha': 0.1,
          'discount_factor': 0.99,
          'epsilon': 1,
          'min_epsilon': 0.25,
          'decay': 0.95,
          'train_episodes': 5000,
          'max_steps': 100,
          'batch_size': 128,
          'replay_buffer_size': 256,
          'target_update': 16
          }

dqn = DQN(**params)
env = gym.make('CartPole-v0')
dqn.learn_environment(env)

for i_episode in range(1):
    state = env.reset().reshape(1, 4)
    for t in range(100):
        env.render()
        print(state)
        action = dqn.act(state)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()
