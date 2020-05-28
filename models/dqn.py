from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, PReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.python.client import device_lib
from keras.callbacks import ModelCheckpoint # this happened to work instead of tf.keras, still waiting for support #https://github.com/keras-team/keras/issues/13258

import numpy as np
import os
from typing import List

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

from environment import DirectReinforcement


def init_dqn(env: DirectReinforcement) -> None:
    tf.keras.backend.clear_session()
    tf.random.set_seed(env.SEED)

    model = deep_q_network(env)
    memory = SequentialMemory(limit=env.mem_size, window_length=env.window_length)
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(), 
        attr='eps',
        value_max=1.0, 
        value_min=0.1, 
        value_test=0.05, 
        nb_steps=env.explore_steps
    )
    nb_actions = env.action_space.n  # set up number of actions (outputs)
    dqn = DQNAgent(
        model=model, 
        gamma=env.gamma, 
        nb_actions=nb_actions, 
        memory=memory,
        batch_size=env.batch_size, 
        nb_steps_warmup=1000,
        target_model_update=env.tar_mod_up, 
        policy=policy, 
        delta_clip=env.delta_clip
    )

    dqn.compile(
        Adam(lr=env.learning_rate, decay=env.learning_rate_decay), 
        metrics=['mse']
    )

    if env.TRAIN_W_VALIDATION:
        train_w_validation(env, dqn)
    else:
        train(env, dqn)


def get_available_cpus() -> List:
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type in ['CPU', 'XLA_CPU']]

def deep_q_network(env: DirectReinforcement) -> Sequential:
    model = Sequential()
    model.add(Flatten(input_shape=(env.window_length,) + env.observation_space.shape))
    model.add(Dense(env.nodes))
    model.add(PReLU())
    model.add(Dense(env.nodes * 2))
    model.add(PReLU())
    model.add(Dense(env.nodes * 4))
    model.add(PReLU())
    model.add(Dense(env.nodes * 2))
    model.add(PReLU())
    model.add(Dense(env.action_space.n))
    model.add(Activation('linear'))

    print('Deep Q Network:')
    print(model.summary())
    
    model_json = model.to_json()

    if not os.path.exists(env.folder):
        os.mkdir(env.folder)
    with open(env.folder + '/model.json', "w") as json_file:
        json_file.write(model_json)

    return model

def train(env: DirectReinforcement, dqn) -> None:
    cpu_devices = get_available_cpus()
    with tf.device(cpu_devices[1]):
        dqn.fit(
            env, 
            nb_steps=env.epochs*env.steps, 
            nb_max_episode_steps=env.steps, 
            visualize=False, 
            verbose=2
        )
        dqn.save_weights(env.folder + '/weights_epoch_{}.h5f'.format(env.epochs), overwrite=True)

    env._plot_actions()
    env._calculate_pnl(env_name=env.env_name)
    np.save(env.folder + '/memory.npy', env.memory)
    env._plot_train_rewards()
    with open(env.folder + '/train_rewards.out', "w") as text_file:
        text_file.write(str(env.rewards))

def train_w_validation(env, dqn):
    filepath = env.folder + '/validate/epochs/'
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    checkpointer = ModelCheckpoint(
        filepath=filepath + 'weights.{epoch:02d}.hdf5', 
        monitor='val_loss', 
        verbose=1,
        save_best_only=False, 
        save_weights_only=True, 
        mode='auto', 
        period=1
    )

    cpu_devices = get_available_cpus()
    with tf.device(cpu_devices[1]):
        dqn.fit(
            env, 
            nb_steps=env.epochs*env.steps, 
            nb_max_episode_steps=env.steps,
            visualize=False, 
            verbose=2, 
            callbacks=[checkpointer]
        )

        env._plot_actions()  # plot last epochs training actions
        env._calculate_pnl(env_name=env.env_name) 
        np.save(env.folder + '/memory.npy', env.memory)
        env._plot_train_rewards()
        env.validation = True

        best_epoch = ""
        best_reward = -1000000
        count_ep = 0

        # iterate through each epoch to find the one with the highest reward
        for weights_file in os.listdir(filepath):

            if weights_file.endswith(".hdf5"):
                count_ep += 1
                print(str(count_ep) + ": Loading: " + weights_file)
                dqn.load_weights(filepath + weights_file)

                env.rewards = []
                env.pnls = []
                env.val_starts_index = 0
                dqn.test(env, nb_episodes=env.val_epochs, nb_max_episode_steps=env.val_steps, visualize=False)

                epoch_rewards = np.sum(env.rewards) / float(env.val_epochs)
                if epoch_rewards > best_reward:
                    best_epoch = weights_file
                    best_reward = epoch_rewards
                    print("BEST EPOCH: " + best_epoch + " with: " + str(best_reward))

        path = filepath + best_epoch
        new_path = env.folder + '/' + best_epoch
        os.rename(path, new_path)
        print("Loading: " + new_path)
        dqn.load_weights(new_path)

        env.validation = False

def test(env: DirectReinforcement, dqn: DQNAgent) -> None:
    cpu_devices = get_available_cpus()
    with tf.device(cpu_devices[1]):
        env.testing = True
        for x in range(env.test_epochs):
            dqn.test(
                env, 
                nb_episodes=1, 
                nb_max_episode_steps=env.test_steps, 
                visualize=False
            )

            env._calculate_pnl(env_name=env.env_name)

            if not os.path.exists(env.test_folder + '/memory_' + str(env.test_starts_index) + '.npy'):
              np.save(env.test_folder + '/memory_' + str(env.test_starts_index) + '.npy', env.memory)
              
            env._plot_actions()
        describe_stats(env, env.test_steps)

def describe_stats(env: DirectReinforcement, steps: int) -> None:
    longs = len(env.long_actions)
    shorts = len(env.short_actions)
    neutrals = steps - longs - shorts
    print("STATS: Long: ", longs , " Short: ", shorts , " Neutral: ", neutrals, " out of ", steps)
