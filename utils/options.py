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
        self.timestamp  = "18050800"    # "yymmdd##"
        # training configuration
        self.mode       = 1             # 1(train) | 2(test model_file)
        self.config     = 1

        self.agent_type, self.env_type, self.game, self.memory_type, self.model_type = CONFIGS[self.config]

        self.seed       = 100
        self.render     = False         # whether render the window from the original envs or not
        self.visualize  = True          # whether do online plotting and stuff or not
        self.save_best  = False         # save model w/ highest reward if True, otherwise always save the latest model

        self.num_envs_per_actor = 1     # NOTE: must be 1 for envs that don't have parallel support
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

        # for preprocessing the states before outputing from env
        if "mlp" in self.model_type:    # low dim inputs, no preprocessing or resizing
            self.state_cha = 1
            self.state_hei = 1          #
            self.state_wid = None       # depends on the env
        elif "cnn" in self.model_type:  # raw image inputs, need to resize or crop to this step_size
            self.state_cha = 1
            self.state_hei = 48         # TODO:
            self.state_wid = 48         # TODO:

        if self.env_type == "gym":
            self.gym_log_dir = None     # when not None, log will be recoreded by baselines monitor
            if self.game == "Pendulum-v0": #  https://gym.openai.com/evaluations/eval_y44gvOLNRqckK38LtsP1Q/
                self.early_stop = 2000  # max #steps per episode
            else:
                self.early_stop = None


class MemoryParams(Params):
    def __init__(self):
        super(MemoryParams, self).__init__()

        if self.memory_type == "shared":
            self.memory_size = 100#1e6


class ModelParams(Params):
    def __init__(self):
        super(ModelParams, self).__init__()

        # NOTE: the devices cannot be passed into the processes this way
        # if 'discrete' in self.model_type:
        #     self.model_device = torch.device('cpu')
        # if 'continuous' in self.model_type:
        #     self.model_device = torch.device('cpu')


class AgentParams(Params):
    def __init__(self):
        super(AgentParams, self).__init__()

        if 'discrete' in self.agent_type:
            # criteria and optimizer
            self.value_criteria = nn.MSELoss()
            self.optim = SharedAdam
        elif 'continuous' in self.agent_type:
            # criteria and optimizer
            self.value_criteria = nn.MSELoss()
            self.optim = SharedAdam
            # generic hyperparameters
            self.num_tasks           = 1    # NOTE: always put main task at last
            self.steps               = 20   # max #iterations
            self.gamma               = 0.99
            self.clip_grad           = 0.5#np.inf
            self.lr                  = 1e-4
            self.lr_decay            = False
            self.weight_decay        = 0.
            self.eval_freq           = 1000#00  # NOTE: here means every this many steps
            self.eval_steps          = 1000
            self.prog_freq           = self.eval_freq
            self.test_nepisodes      = 50
            # off-policy specifics
            self.learn_start         = 25   # start update params after this many steps
            self.batch_size          = 16
            self.valid_size          = 500
            self.eps_start           = 1
            self.eps_end             = 0.1
            self.eps_eval            = 0.#0.05
            self.eps_decay           = 1000000
            self.target_model_update = 1e-3#1000
            self.action_repetition   = 4
            self.memory_interval     = 1
            self.train_interval      = 4


class Options(Params):
    env_params = EnvParams()
    memory_params = MemoryParams()
    model_params = ModelParams()
    agent_params = AgentParams()
