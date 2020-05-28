from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 

np.set_printoptions(precision=8, suppress=True)

import gym
import typing

from lib.visualization import plot_profit, plot_actions, plot_train_rewards

import math
import os
import random

from sklearn.cluster import KMeans

class DirectReinforcement(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, *args, **kwargs):
        def init_cfg(cfg):
            for _field in cfg._fields:
                setattr(self, _field, getattr(cfg, _field))
        for arg in args:
            init_cfg(arg)

        np.random.seed(self.SEED)
        random.seed(self.SEED)

        data = np.load(self.train_data, allow_pickle=True)
        self.price_data = data[:,0]
        self.data_size = len(self.price_data)
        self.return_std = np.std(self.price_data[1:]-self.price_data[:-1])
        self.news_sentiment = data[:,1]
        self.news_embeddding = data[:,2]
        self.eps_data = data[:,3]

        self.scaling_factor = {
            'return_std': np.std(self.price_data[1:]-self.price_data[:-1]),
            'return_mean': np.mean(self.price_data[1:]-self.price_data[:-1]),
            'eps_std': np.std([elem for elem in self.eps_data if elem != 0]),
            'eps_mean': np.mean([elem for elem in self.eps_data if elem != 0]),
        }

        input_shape = self.window
        input_shape *= self.no_of_cluster
        input_shape += self.window * self.sentimental_feature
        input_shape += self.window * self.eps_feature
        if self.embedding_feature:
            input_shape += self.embedding_feature_len
        if self.action_one_hot:
            input_shape += 3
        print("Input shape:", input_shape)

        self.observation_space = gym.spaces.Box(
            low=-1,
            high=2,
            shape=(input_shape,)
        )
        self.action_space = gym.spaces.Discrete(3)

        self.memory = []
        self.testing = False
        self.validation = False
        self.rewards = []
        self.epoch_reward = 0
        self.epoch_profit = []

        self.test_starts_index = 0
        self.val_starts_index = 0

        self.reset()

    def step(self, action):
        """
        Take action, move agent to next position and make a trade action
        Store the actiona and the new value
        Get reward

        Return: new state, reward and whether the data is done
        """
        c_val = self.price_data[self.position]

        if action == 2: # sell / short:
            self.action = self.SELL
            self.short_actions.append([self.position, c_val])
        elif action == 1 : # buy / long
            self.action = self.BUY
            self.long_actions.append([self.position, c_val])
        else:
            self.action = self.NEUTRAL

        if (self.position+1) < self.data_size:
            state = [self.position, c_val, self.action]
            self.memory.append(state)

            self.position += 1
            self.reward = self._get_reward()
            self.epoch_reward += self.reward
            self.epoch_profit.append(self.reward)
            self.observation = self._next_observation_input()
        else:
            self.done = True

        return self.observation, self.reward, self.done, {}      

    def reset(self):
        if self.testing:
            data = np.load(self.test_data, allow_pickle=True)
            self.price_data = data[:,0]
            self.data_size = len(self.price_data)
            self.return_std = np.std(self.price_data[1:]-self.price_data[:-1])
            self.news_sentiment = data[:,1]
            self.news_embeddding = data[:,2]
            self.eps_data = data[:,3]
            self.test_position = np.random.randint(self.window + 1, self.data_size - self.test_steps - 1, self.test_epochs)
            self.position = self.test_position[self.test_starts_index]
            self.test_starts_index += 1
            self.test_folder = self.folder + '/Test_' + str(self.test_starts_index)
            if not os.path.exists(self.test_folder):
                os.makedirs(self.test_folder)
        elif self.validation:
            data = np.load(self.val_data, allow_pickle=True)
            self.price_data = data[:,0]
            self.data_size = len(self.price_data)
            self.return_std = np.std(self.price_data[1:]-self.price_data[:-1])
            self.news_sentiment = data[:,1]
            self.news_embeddding = data[:,2]
            self.eps_data = data[:,3]
            self.val_position = np.random.randint(self.window + 1, self.data_size - self.val_steps - 1, size=self.val_epochs)
            self.position = self.val_position[self.val_starts_index]
            self.val_starts_index += 1
        else:
            begin_idx = self.window + 1
            end_idx = self.data_size - self.steps - 1
            self.position = random.randint(begin_idx, end_idx)

        self.memory = []
        self.long_actions = []
        self.short_actions = []
        self.trades = []
        self.long_prec = 0
        self.short_prec = 0
        self.reward = 0
        self.rewards.append(self.epoch_reward)
        self.action = 0
        self.prev_action = 0
        self.buy_flag = False
        self.sell_flag = False
        self.done = False
        self.observation = self._next_observation_input()

        return self.observation

    def render(self, mode='human', close=False):
        """
        Gym function called at the end of a process.
        """
        self._calculate_pnl(env_name=self.env_name, save=False)
        self.reset()
        return None

    def _calculate_pnl(self, env_name, save=True):
        """
        Calculate the final PnL based on the actions of the agent with three different fee values (slippage)
        """
        actions = np.array([x[2] for x in self.memory])
        values = np.array([x[1] for x in self.memory]).reshape((-1,))

        self.pnl = self._pnl_of_trades(env_name, actions, values)

        pnls = "Profit and loss: " + str(format(self.pnl, '.5f')) + "\n"
        if len(self.long_actions) != 0:
            l_prec = str(format((self.long_prec / float(len(self.long_actions))), '.2f'))
        else:
            l_prec = str(0)
        longs = str(len(self.long_actions))
        pnls += "Precision Long: " + l_prec + " ("+ str(self.long_prec) + " of " + longs + ")\n"

        if len(self.short_actions) != 0:
            s_prec = str(format((self.short_prec / float(len(self.short_actions))), '.2f'))
        else:
            s_prec = str(0)

        shorts = str(len(self.short_actions))
        pnls += "Precision Short: " + s_prec + " ("+ str(self.short_prec) + " of " + shorts + ")\n"
        pnls += "Test reward: " + str(self.epoch_reward) + "\n"

        print(pnls)

        if save:
            if self.testing:
                file_sl = '/test_pnl_' + str(self.test_starts_index) + '.out'
                with open(self.test_folder + file_sl, "w") as text_file:
                    text_file.write(pnls)
                file_tr = '/test_trades_' + str(self.test_starts_index) + '.out'
                with open(self.test_folder + file_tr, "w") as text_file:
                    text_file.write(str(self.trades))
            else:
                file_sl = '/train_pnl.out'
                with open(self.folder + file_sl, "w") as text_file:
                    text_file.write(pnls)
                file_tr = '/train_trades.out'
                with open(self.folder + file_tr, "w") as text_file:
                    text_file.write(str(self.trades))

    def _pnl_of_trades(self, env_name, actions, values):
        prices_diff = np.concatenate([np.diff(values), [0.0]])

        correct_actions = []
        action = actions[0]
        for i in range(len(actions)):
            if actions[i] != self.NEUTRAL:
                correct_actions.append(actions[i])
                action = actions[i]
            else:
                correct_actions.append(action)
        pnl = np.cumsum([correct_actions*prices_diff])
        plot_profit(self.folder, pnl, np.cumsum(self.epoch_profit), values, actions)
        return (pnl[-1])

    def _next_observation_input(self):
        """
        Prepare input to the agent
        """
        z_returns = []
        s_sentiments = []
        e_eps = []

        idx = self.position
        n_news_embedding = self.news_embeddding[idx]

        for i in range(self.window):
            c_val = self.price_data[idx]
            pr_val = self.price_data[idx-1]
            z_return = ((c_val - pr_val) - self.scaling_factor['return_mean']) / self.scaling_factor['return_std']
            z_returns.append(z_return)

            news = self.news_sentiment[idx]
            s_sentiments.append(news)

            eps = self.eps_data[idx]
            if eps != 0:
                eps = (self.eps_data[idx] - self.scaling_factor['eps_mean']) / self.scaling_factor['eps_std']
            e_eps.append(eps)

            idx -= 1

        obs = np.asarray(z_returns, dtype=np.float32)
        obs = self._fuzzy_representation(obs, self.no_of_cluster)
        obs = np.append(obs, s_sentiments)
        obs = np.append(obs, e_eps)
        obs = np.append(obs, n_news_embedding)

        if self.action_one_hot:
            a = int(self.action == self.BUY)
            b = int(self.action == self.NEUTRAL)
            c = int(self.action == self.SELL)
            obs = np.append(obs, [a,b,c])

        return obs

    def _fuzzy_representation(self, input_s_zts, n_clusters):
        input_s_zts = input_s_zts.reshape(-1,1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=17061996).fit(input_s_zts)
        cluster_labels = kmeans.labels_
        center = kmeans.cluster_centers_

        def __calculate_mf(data, center, cluster_label):
            mf_ls = []
            def ___gauss_mf(x):
                return math.exp(-(mean-x)**2/(2*(var)**2))
            for i, (centroid_x) in enumerate(center):
                mean = centroid_x 
                var = np.sum([(x-mean)**2 for x in data[cluster_label==i]])/len(data[cluster_label==i])+0.0000000001
                data_point = [point for point in data]
                mf_with_curr_centroid = list(map(___gauss_mf, data_point))
                mf_ls.append(mf_with_curr_centroid)
            return np.array(mf_ls).T.reshape(-1,1)

        mf_val = __calculate_mf(input_s_zts, center, cluster_labels)
        mf_val = mf_val.reshape(len(mf_val),)
        return mf_val

    def _get_reward(self):
        c_val = self.price_data[self.position]
        pr_val = self.price_data[self.position-1]

        dt = self.action
        zt = (c_val - pr_val)
        c = self.cost * c_val
        m = np.abs(dt - self.prev_action)

        reward = dt*zt - c*m
        reward = reward / self.REWARD_STD 

        self._calc_precision(c_val, pr_val)
        self._trade(c_val)
        self.prev_action = self.action

        return reward

    def _calc_precision(self, c_val, pr_val):
        """
        Calculate if the actions taken by the agent were indeed correct
        """
        if self.prev_action == self.BUY: # buy / long
            if c_val > pr_val or c_val == pr_val:
                self.long_prec += 1
        elif self.prev_action == self.SELL:
            if c_val < pr_val or c_val == pr_val:
                self.short_prec += 1

    def _trade(self, c_val):
        """
        Save that a trade has been made at the current time step
        """
        if self.action == self.BUY:
            if not self.buy_flag:
                self.sell_flag = False
                self.buy_flag = True
                self.trades.append([self.position, c_val])
        elif self.action == self.SELL:
            if not self.sell_flag:
                self.sell_flag = True
                self.buy_flag = False
                self.trades.append([self.position, c_val])

    def _plot_actions(self):
        plot_actions(self.folder, self.memory, self.long_actions, self.short_actions)

    def _plot_train_rewards(self):
        plot_train_rewards(self.folder, self.rewards)

