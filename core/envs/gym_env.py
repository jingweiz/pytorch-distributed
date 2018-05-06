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


class GymEnv(Env):
    def __init__(self, args, process_ind=0, num_envs_per_process=1):
        super(GymEnv, self).__init__(args, process_ind, num_envs_per_process)

        # env_params for this env
        self.gym_log_dir = args.gym_log_dir

        # init envs
        print("env config ----------->", "process_ind:", str(process_ind), "num_envs_per_process:", str(num_envs_per_process))
        for i in range(self.num_envs_per_process):
            print("env seed --->", str(args.seed + self.process_ind*self.num_envs_per_actor + i))
        envs = [make_env(args.game, args.seed, self.process_ind*self.num_envs_per_actor+i, self.gym_log_dir)
                for i in range(self.num_envs_per_process)]

        if self.num_envs_per_process > 1:
            self.env = SubprocVecEnv(envs)
        else:
            #self.env = DummyVecEnv(envs) NOTE double check
            self.env = envs[0]()

        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high

    def _preprocessState(self, state):
        return state # TODO

    @property
    def state_shape(self):
        if len(self.env.observation_space.shape)<2:
            return [self.env.observation_space.shape[0],1]
        else:
            return self.env.observation_space.shape

    @property
    def action_shape(self):
        return self.env.action_space.shape[0]

    def step(self, action):
        self.exp_action = action
        execute_action = np.clip(action*self.action_high,
                self.action_low,
                self.action_high)
        self.exp_state1, self.exp_reward, self.exp_terminal1, _ = self.env.step(execute_action)
        return self._get_experience()

    def reset(self):
        self._reset_experience()
        self.exp_state1 = self.env.reset().reshape(-1,1)
        return self._get_experience()
