import numpy as np
from collections import deque
import atari_py
# import torch
import cv2  # Note that importing cv2 before torch may cause segfaults?

from utils.helpers import Experience            # NOTE: here state0 is always "None"
from core.env import Env

class AtariEnv(Env):
    def __init__(self, args, process_ind=0, num_envs_per_process=1):
        super(AtariEnv, self).__init__(args, process_ind, num_envs_per_process)

        # env_params for this env
        assert self.num_envs_per_process == 1
        self.seed = self.seed + self.process_ind * self.num_envs_per_actor  # NOTE: check again

        # setup ale
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', self.seed)
        self.ale.setInt('max_num_frames', self.early_stop)
        self.ale.setFloat('repeat_action_probability', 0)   # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        print(atari_py.get_game_path(self.game))
        self.ale.loadROM(atari_py.get_game_path(self.game)) # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))

        self.lives = 0          # life counter (used in DeepMind training)
        self.just_died = False  # when lost one life, but game is still not over

        # setup
        self.exp_state1 = deque(maxlen=self.state_cha)
        self._reset_experience()

    def _reset_experience(self):
        self.exp_state0 = None  # NOTE: always None in this module
        self.exp_action = None
        self.exp_reward = None
        for _ in range(self.state_cha):
            # self.exp_state1.append(torch.zeros(self.state_hei, self.state_wid))
            self.exp_state1.append(np.zeros((self.state_hei, self.state_wid), dtype=np.uint8))
        self.exp_terminal1 = None

    def _get_experience(self):
        return Experience(state0 = self.exp_state0, # NOTE: here state0 is always None
                          action = self.exp_action,
                          reward = self.exp_reward,
                          state1 = self._preprocess_state(self.exp_state1),
                          terminal1 = self.exp_terminal1)

    def _capture_ale(self):
        # NOTE: the gray scale state comes in [0, 255] (210, 160, 1) already, in np.uint8 !!!
        # NOTE: returned image also        in [0, 255] (84, 84),              in np.uint8 !!!
        return cv2.resize(self.ale.getScreenGrayscale(),
                          (self.state_hei, self.state_wid),
                          interpolation=cv2.INTER_LINEAR)

    def _preprocess_state(self, state):
        return np.stack(state, axis=0)

    def _preprocess_action(self, action):
        return action.reshape(self.action_shape) # NOTE: here using action_shape instead of action_space

    @property
    def norm_val(self):  # NOTE: the max value of states, use this to normalize model inputs
        return 255.

    @property
    def state_shape(self):  # NOTE: here returns the shape after preprocessing, i.e., the shape that gets passed out that's pushed into memory
        return [self.state_cha, self.state_hei, self.state_wid]

    @property
    def action_shape(self):
        return 1

    @property
    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

    def step(self, action):
        self.exp_action = self._preprocess_action(action)
        # repeat action 4 times, max pool over last 2 frames
        frame_buffer = np.zeros((2, self.state_hei, self.state_wid), dtype=np.uint8)
        self.exp_reward = 0.
        self.exp_terminal1 = False
        for t in range(4):
            self.exp_reward += self.ale.act(self.actions.get(action[0][0]))
            if t == 2:
                frame_buffer[0] = self._capture_ale()
            elif t == 3:
                frame_buffer[1] = self._capture_ale()
            self.exp_terminal1 = self.ale.game_over()
            if self.exp_terminal1:
                break
        last_state = np.maximum(frame_buffer[0], frame_buffer[1])
        self.exp_state1.append(last_state)
        # detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:        # lives > 0 for Q*bert TODO: Qbert???
                self.just_died = not self.exp_terminal1 # only set flag when not truly done
                self.exp_terminal1 = True
            self.lives = lives
        return self._get_experience()

    def reset(self):
        if self.just_died:
            self.just_died = False
            self.ale.act(0) # use a no-op after loss of life
        else:
            self._reset_experience()
            self.ale.reset_game()
            # perform up to 30 random no-ops before starting
            for _ in range(np.random.randint(30)):
                self.ale.act(0)  # assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        self.exp_state1.append(self._capture_ale())
        self.lives = self.ale.lives()
        return self._get_experience()
