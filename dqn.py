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
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam, RMSprop
from tqdm import tqdm

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'is_terminal'))

seed_value = 42
def init_seeds():
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    
init_seeds()

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



def get_model(input_shape=tuple([4]), action_space=2, learning_rate=0.001, hidden_layers = (32,32,32), verbose=False):
    X_input = Input(input_shape)
    
    for index, hl in enumerate(hidden_layers):
        if index == 0:
            X = Dense(hl, activation="relu")(X_input)    
        else:
            X = Dense(hl, activation="relu")(X)
            

    X = Dense(action_space, activation="linear")(X)
    model = Model(inputs=X_input, outputs=X, name='CartPole_DQN_model')
    model.compile(loss="mse", optimizer=Adam(learning_rate=learning_rate), metrics=["accuracy"])
    
    if verbose:
        model.summary()

    return model


class DQN:
    def __init__(self, alpha: float, discount_factor: float, epsilon: float, min_epsilon: float, decay: float,
                 train_episodes: int, max_steps: int, batch_size: int,
                 replay_buffer_size: int, target_update: int, result_folder: str, train_start: int, layer_struct):
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
        self.target_model = get_model(learning_rate = alpha, hidden_layers=layer_struct, verbose=True)
        self.Q = get_model(learning_rate = alpha, hidden_layers=layer_struct)
        self.result_folder = result_folder
        self.train_start = train_start
        self.history = None

    def act(self, state):
        return np.argmax(self.Q.predict(state)[0])

    def learn_environment(self, env, verbose=False):
        self.update_target_model()
        epsilon = self.epsilon
        acc_rewards = []
        epsilons = []
        losses = []
        episode = -1
        threshold_for_stopping = 475
        
        for _ in tqdm(range(self.train_episodes)):
            

            
            episode += 1
            
            self.save_model()
            self.save_list(acc_rewards, 'rewards')
            self.save_list(epsilons, 'epsilons')
            self.save_list([episode], 'am_alive')
                
            if self.history is not None:
                losses += list(self.history.history['loss'])
                self.save_list(losses, 'loss')
                
            if len(acc_rewards) > 100 and np.mean(acc_rewards[-100:]) > threshold_for_stopping:
                break

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
                

                if ((self.replay_counter + 1) % self.target_update_counter) == 0:
                    self.update_target_model()

                state = new_state

                if terminal:
                    break
                    
            self.replay_training()
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

        self.history = self.Q.fit(states, target, batch_size=self.batch_size, verbose=False)
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


res_folder = 'res3'
params = {'alpha': 0.001,
          'discount_factor': 0.95,
          'epsilon': 1,
          'min_epsilon': 0.01,
          'decay': 0.995,
          'train_episodes': 10000,
          'max_steps': 500,
          'batch_size': 64,
          'replay_buffer_size': 16000,
          'target_update': 3,
          'result_folder': res_folder,
          'train_start': 128,
          'layer_struct': (32,32,32, 16,8)
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
