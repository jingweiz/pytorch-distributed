import numpy as np
import os
import shutil
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn

from optims.sharedAdam import SharedAdam
from optims.sharedRMSprop import SharedRMSprop

CONFIGS = [
# agent_type,   env_type, game,          memory_type, model_type
[ "discrete",   "empty",  "",            "shared",    "discrete-mlp"  ], # 0 *
[ "continuous", "gym",    "Pendulum-v0", "shared",    "continuous-mlp"], # 1 *
]

class Params(object):
    def __init__(self):
        # training signature
        self.machine    = "aisdaim"     # "machine_id"
        self.timestamp  = "18042600"    # "yymmdd##"
        # training configuration
        self.mode       = 1             # 1(train) | 2(test model_file)
        self.config     = 1

        self.agent_type, self.env_type, self.game, self.memory_type, self.model_type = CONFIGS[self.config]

        self.seed       = 123
        self.render     = False         # whether render the window from the original envs or not
        self.visualize  = True          # whether do online plotting and stuff or not
        self.save_best  = False         # save model w/ highest reward if True, otherwise always save the latest model

        self.num_envs_per_actor = 4     # NOTE: must be 1 for envs that don't have parallel support
        self.num_actors = 2
        self.num_learners = 1

        # prefix for saving models&logs
        self.refs       = self.machine + "_" + self.timestamp
        self.root_dir   = os.getcwd()

        # model files
        # NOTE: will save the current model to model_name
        self.model_name = self.root_dir + "/models/" + self.refs + ".pth"
        # NOTE: will load pretrained model_file if not None
        self.model_file = None#self.root_dir + "/models/{TODO:FILL_IN_PRETAINED_MODEL_FILE}.pth"
        if self.mode == 2:
            self.model_file = self.model_name  # NOTE: so only need to change self.mode to 2 to test the current training
            assert self.model_file is not None, "Pre-Trained model is None, Testing aborted!!!"
            self.visualize = False

        # logging configs
        self.log_dir = self.root_dir + "/logs/" + self.refs + "/"


class EnvParams(Params):
    def __init__(self):
        super(EnvParams, self).__init__()


class MemoryParams(Params):
    def __init__(self):
        super(MemoryParams, self).__init__()


class ModelParams(Params):
    def __init__(self):
        super(ModelParams, self).__init__()


class AgentParams(Params):
    def __init__(self):
        super(AgentParams, self).__init__()


class Options(Params):
    env_params = EnvParams()
    memory_params = MemoryParams()
    model_params = ModelParams()
    agent_params = AgentParams()
