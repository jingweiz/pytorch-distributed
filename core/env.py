from utils.helpers import Experience            # NOTE: here state0 is always "None"
import gym


class Env(object):
    def __init__(self, args, process_ind=0, num_envs_per_process=1):
        # self.board = args.board
        self.process_ind = process_ind      # NOTE: this just to ensure we have diff seeds for diff envs
        self.num_envs_per_process = num_envs_per_process    # NOTE: to make sure diff seeds for each env, bit tricky
        self.num_envs_per_actor = args.num_envs_per_actor   # NOTE: to make sure diff seeds for each env, bit tricky

        # params
        self.mode       = args.mode         # NOTE: save frames when mode=2
        self.seed       = args.seed         # NOTE: so to give a different seed to each instance
        # self.visualize  = args.visualize    # TODO: setup tensorboard stuff

        # env_params
        self.env_type   = args.env_type
        self.game       = args.game

        self.state_cha = args.state_cha
        self.state_hei = args.state_hei
        self.state_wid = args.state_wid

        # setup
        self._reset_experience()

    def _reset_experience(self):
        self.exp_state0 = None  # NOTE: always None in this module
        self.exp_action = None
        self.exp_reward = None
        self.exp_state1 = None
        self.exp_terminal1 = None

    def _get_experience(self):
        return Experience(state0 = self.exp_state0, # NOTE: here state0 is always None
                          action = self.exp_action,
                          reward = self.exp_reward,
                          state1 = self._preprocess_state(self.exp_state1),
                          terminal1 = self.exp_terminal1)

    def _preprocess_state(self, state):
        raise NotImplementedError("not implemented in base class")

    @property
    def state_shape(self):
        raise NotImplementedError("not implemented in base class")

    @property
    def action_shape(self):
        raise NotImplementedError("not implemented in base class")

    def render(self):       # render using the original gl window
        raise NotImplementedError("not implemented in base class")

    def visual(self):       # visualize onto tensorboard
        raise NotImplementedError("not implemented in base class")

    def reset(self):
        raise NotImplementedError("not implemented in base class")

    def step(self, action):
        raise NotImplementedError("not implemented in base class")
