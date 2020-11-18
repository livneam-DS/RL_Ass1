from collections import namedtuple
import tensorflow as tf
import os
from gym.envs.registration import EnvRegistry
from keras.models import Sequential
import numpy as np
from keras.layers import Dense, Lambda, Add
import gym
import random
import json
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam, RMSprop
from tqdm import tqdm
import keras.backend as K
from itertools import product

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'is_terminal'))

seed_value = 42


def init_seeds():
    os.environ['PYTHONHASHSEED'] = str(seed_value)
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


def get_model(input_shape=tuple([4]), action_space=2, learning_rate=0.001, hidden_layers=(32, 32, 32), verbose=False,
              dueling=False):
    X_input = Input(input_shape)
    X = None

    for index, hl in enumerate(hidden_layers):
        if index == 0:
            X = Dense(hl, activation="relu")(X_input)
        else:
            X = Dense(hl, activation="relu")(X)

    if not dueling:
        X = Dense(action_space, activation="linear")(X)
    else:
        state_value = Dense(1)(X)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(action_space,))(state_value)

        action_advantage = Dense(action_space)(X)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_space,))(
            action_advantage)
        X = Add()([state_value, action_advantage])

    model = Model(inputs=X_input, outputs=X, name='CartPole_DQN_model')
    model.compile(loss="mse", optimizer=Adam(learning_rate=learning_rate), metrics=["accuracy"])

    if verbose:
        model.summary()

    return model


class DQN:
    def __init__(self, alpha: float, discount_factor: float, epsilon: float, min_epsilon: float, decay: float,
                 train_episodes: int, max_steps: int, batch_size: int,
                 replay_buffer_size: int, target_update: int, result_folder: str, train_start: int, layer_struct: tuple,
                 dueling: bool):
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
        self.target_model = get_model(learning_rate=alpha, hidden_layers=layer_struct, verbose=True, dueling=dueling)
        self.Q = get_model(learning_rate=alpha, hidden_layers=layer_struct, dueling=dueling)
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
        step_counter = 0

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
                step_counter += 1
                
                new_state = new_state.reshape(1, 4)
                current_reward += reward

                if not terminal or step == env._max_episode_steps - 1:
                    reward = reward
                else:
                    reward = -500

                self.replay_buffer.push(state, action, new_state, reward, terminal)
                self.replay_training()

                if ((step_counter + 1) % self.target_update_counter) == 0:
                    self.update_target_model()

                state = new_state

                if terminal:
                    break
                epsilon = max(self.min_epsilon, self.decay * epsilon)
            
            acc_rewards.append(current_reward)
            
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



                

def_params = {'alpha': 0.00005,
          'discount_factor': 0.95,
          'epsilon': 1,
          'min_epsilon': 0.001,
          'decay': 0.995,
          'train_episodes': 10000,
          'max_steps': 500,
          'batch_size': 128,
          'replay_buffer_size': 2 ** 14,
          'target_update': 250,
          'result_folder': 'testDDQN/',
          'train_start': 1024,
          'layer_struct': (32, 32, 32),
          'dueling': False
         }

dqn = DQN(**def_params)
env = gym.make('CartPole-v1')
dqn.learn_environment(env)
dqn.save_model()

# lr = [0.001, 0.005]
# batch_size = [64,128]
# decay = [0.995, 0.95]
# target_update = [3, 5]
# layers_struct = [(32, 32, 32, 16, 8), (32, 32, 32)]
# dueling = [True]

# params_to_run = list(product(*[lr, batch_size, decay, target_update,layers_struct, dueling]))
# for index, param_combination in enumerate(params_to_run):
#     lr_val, batch_val, decay_val, target_update_val, layers_struct_val, dueling_val = param_combination
    
#     param_dict = def_params.copy()
#     param_dict['alpha'] = lr_val
#     param_dict['batch_size'] = batch_val
#     param_dict['decay'] = decay_val
#     param_dict['target_update'] = target_update_val
#     param_dict['layer_struct'] = layers_struct_val
#     param_dict['dueling'] = dueling_val
    
    
#     if dueling_val:
#         res_folder = f'iterative_runs/with_deuling/run_{index}'
#     else:
#         res_folder = f'iterative_runs/without/run_{index}'
    
    
#     param_dict['result_folder'] = res_folder
    
#     try:
#         os.mkdir(res_folder)
#     except:
#         pass

#     with open(f'{res_folder}/expr_params.json', 'w+') as f:
#         json.dump(param_dict, f)

#     dqn = DQN(**param_dict)
#     env = gym.make('CartPole-v1')
#     dqn.learn_environment(env)
#     dqn.save_model()
