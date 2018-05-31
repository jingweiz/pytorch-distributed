#!/usr/bin/env python
# coding=utf-8
'''
Author:Tai Lei
Date:Wednesday, April 11, 2018 PM09:29:12 HKT
Info: This is the wrapper for envs created from normal env
'''

import numpy as np
import sys
import gym  # TODO: import inside

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from utils.helpers import Experience            # NOTE: here state0 is always "None"
from core.env import Env
from core.envs.make_env import make_env
from gym.spaces.discrete import Discrete


class GymEnv(Env):
    def __init__(self, args, process_ind=0, num_envs_per_process=1):
        super(GymEnv, self).__init__(args, process_ind, num_envs_per_process)

        # env_params for this env
        self.gym_log_dir = args.gym_log_dir

        # init envs
        # print("env config ---------->", str(process_ind), str(num_envs_per_process))
        # for i in range(self.num_envs_per_process):
            # print("env seed --->", str(args.seed + self.process_ind*self.num_envs_per_actor + i))
        envs = [make_env(args.game, args.seed, self.process_ind*self.num_envs_per_actor+i, self.gym_log_dir)
                for i in range(self.num_envs_per_process)]

        if self.num_envs_per_process > 1:
            self.env = SubprocVecEnv(envs)
        else:
            #self.env = DummyVecEnv(envs) NOTE double check
            self.env = envs[0]()

        if isinstance(self.env.action_space, Discrete):
            self.discrete = True
        else:
            self.discrete = False
            self.action_low = self.env.action_space.low
            self.action_high = self.env.action_space.high

    def _preprocess_state(self, state):
        return state.reshape(self.state_shape) # TODO

    def _preprocess_action(self, action):
        return action.reshape(self.action_shape) # NOTE: here using action_shape instead of action_space

    @property
    def state_shape(self):  # NOTE: here returns the shape after preprocessing, i.e., the shape that gets passed out that's pushed into memory
        if len(self.env.observation_space.shape) < 2:
            # return [self.state_cha, self.state_hei, self.env.observation_space.shape[0]]
            return [self.state_hei, self.env.observation_space.shape[0]]
        else:
            # return self.env.observation_space.shape
            # return [self.state_cha, self.state_hei, self.state_wid]
            return [self.state_hei, self.state_wid]

    @property
    def action_shape(self):
        if self.discrete:
            return 1    # TODO: hardcoded for now
        else:
            return self.env.action_space.shape[0]

    @property
    def action_space(self):
        if self.discrete:
            return self.env.action_space.n
        else:
            return self.env.action_space.shape[0]

    def step(self, action):
        self.exp_action = self._preprocess_action(action)
        if self.discrete:
            execute_action = action
        else:
            execute_action = np.clip(action*self.action_high,
                    self.action_low,
                    self.action_high)
        self.exp_state1, self.exp_reward, self.exp_terminal1, _ = self.env.step(execute_action)
        if self.discrete: # NOTE: somehow gym returns float for discrete, ndarray for continuous
            tmp_reward = self.exp_reward
            self.exp_reward = np.ndarray(1)
            self.exp_reward.fill(tmp_reward)
        return self._get_experience()

    def reset(self):
        self._reset_experience()
        self.exp_state1 = self.env.reset()
        return self._get_experience()
