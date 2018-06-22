import numpy as np
import os
import gym
from gym.spaces.box import Box
#from baselines import bench
#from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from core.envs.atari_wrappers import make_atari, wrap_deepmind

try:
    import pybullet_envs
    import roboschool
except ImportError:
    pass

def make_env(args, rank):
    def _thunk():
        env = gym.make(args.game)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(args.game)
        env.seed(args.seed + rank)
        #if log_dir is not None:
            #env = bench.Monitor(env, os.path.join(args.log_dir, str(rank)))
        if is_atari:
            env = wrap_dqn(env, frame_stack=True)
        return env
    return _thunk
