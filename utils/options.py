import numpy as np
import os
import shutil
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn

from utils.random_process import OrnsteinUhlenbeckProcess
# from optims.sharedAdam import SharedAdam
# from optims.sharedRMSprop import SharedRMSprop

CONFIGS = [
# agent_type, env_type, game,                 memory_type, model_type
[ "dqn",      "gym",    "CartPole-v0",        "shared",    "dqn-mlp" ], # 0
[ "dqn",      "gym",    "Pong-ram-v0",        "shared",    "dqn-mlp" ], # 1
[ "dqn",      "gym",    "PongNoFrameskip-v4", "shared",    "dqn-cnn" ], # 2
[ "ddpg",     "gym",    "Pendulum-v0",        "shared",    "ddpg-mlp"], # 3
]

class Params(object):
    def __init__(self):
        # training signature
        self.machine    = "aisdaim"     # "machine_id"
        self.timestamp  = "18060400"    # "yymmdd##"
        # training configuration
        self.mode       = 1             # 1(train) | 2(test model_file)
        self.config     = 1

        self.agent_type, self.env_type, self.game, self.memory_type, self.model_type = CONFIGS[self.config]

        self.seed       = 100
        self.render     = False         # whether render the window from the original envs or not
        self.visualize  = True          # whether do online plotting and stuff or not

        self.num_envs_per_actor = 1     # NOTE: must be 1 for envs that don't have parallel support
        self.num_actors = 8
        self.num_learners = 1           # TODO: currently have only considered 1 learner; should enable also set each learner to a separate device

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
            self.state_cha = 1          # NOTE: equals hist_len
            self.state_hei = 1          # NOTE: always 1 for mlp's
            self.state_wid = None       # depends on the env
        elif "cnn" in self.model_type:  # raw image inputs, need to resize or crop to this step_size
            self.state_cha = 1          # NOTE: equals hist_len
            self.state_hei = 42
            self.state_wid = 42
        assert self.state_cha == 1      # NOTE: to ease storing into memory

        if self.env_type == "gym":
            self.gym_log_dir = None     # when not None, log will be recoreded by baselines monitor

            # max #steps per episode
            if self.game == "Pendulum-v0":  #  https://gym.openai.com/evaluations/eval_y44gvOLNRqckK38LtsP1Q/
                self.early_stop = 200
            else:                       # for the ataris
                self.early_stop = None


class MemoryParams(Params):
    def __init__(self):
        super(MemoryParams, self).__init__()

        if self.memory_type == "shared":
            if self.agent_type == "dqn":
                self.memory_size = 1000000
            elif self.agent_type == "ddpg":
                self.memory_size = 50000

            self.enable_prioritized = False     # TODO: tbi
            if self.enable_prioritized:
                self.priority_exponent = 0.5    # TODO: taken from rainbow, check for distributed
                self.priority_weight = 0.4      # TODO: taken from rainbow, check for distributed


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

        if self.agent_type == "dqn":
            # criteria and optimizer
            self.value_criteria = nn.MSELoss()
            # self.optim = torch.optim.RMSprop
            self.optim = torch.optim.Adam
            # generic hyperparameters
            self.num_tasks           = 1    # NOTE: always put main task at last
            self.steps               = 1000000 # max #iterations
            self.gamma               = 0.99
            self.clip_grad           = 40.#100
            self.lr                  = 1e-4#2.5e-4/4.
            self.lr_decay            = False
            self.weight_decay        = 0.
            self.actor_sync_freq     = 400  # sync global_model to actor's local_model every this many steps
            # logger configs
            self.logger_freq         = 15   # log every this many secs
            self.actor_freq          = 2500 # push & reset local actor stats every this many actor steps
            self.learner_freq        = 1000 # push & reset local learner stats every this many learner steps
            self.evaluator_freq      = 60   # eval every this many secs
            self.evaluator_steps     = 3000 # eval for this many steps
            self.tester_nepisodes    = 50
            # off-policy specifics
            self.learn_start         = 50000 # start update params after this many steps
            self.batch_size          = 64
            self.target_model_update = 1e-3
            self.hist_len            = 4    # NOTE: each sample state contains this many frames
            self.nstep               = 1
            # dqn specifics
            self.enable_double       = True#False
            self.eps                 = 0.4
            self.eps_alpha           = 7
            self.action_repetition   = 4
        elif self.agent_type == "ddpg":
            # criteria and optimizer
            self.value_criteria = nn.MSELoss()
            self.optim = torch.optim.Adam
            # generic hyperparameters
            self.num_tasks           = 1    # NOTE: always put main task at last
            self.steps               = 1000000 # max #iterations
            self.gamma               = 0.99
            self.clip_grad           = 40.
            self.lr                  = 1e-4
            self.lr_decay            = False
            self.weight_decay        = 0.
            self.actor_sync_freq     = 400  # sync global_model to actor's local_model every this many steps
            # logger configs
            self.logger_freq         = 15   # log every this many secs
            self.actor_freq          = 2500 # push & reset local actor stats every this many actor steps
            self.learner_freq        = 1000 # push & reset local learner stats every this many learner steps
            self.evaluator_freq      = 60   # eval every this many secs
            self.evaluator_steps     = 1000 # eval for this many steps
            self.tester_nepisodes    = 50
            # off-policy specifics
            self.learn_start         = 200  # start update params after this many steps
            self.batch_size          = 64
            self.target_model_update = 1e-3
            self.hist_len            = 1    # NOTE: each sample state contains this many frames
            self.nstep               = 5    # NOTE: this many steps lookahead
            # ddpg specifics
            self.random_process      = OrnsteinUhlenbeckProcess
            self.action_repetition   = 1    # NOTE: just to use the same evaluator & tester as dqn


class Options(Params):
    env_params = EnvParams()
    memory_params = MemoryParams()
    model_params = ModelParams()
    agent_params = AgentParams()
