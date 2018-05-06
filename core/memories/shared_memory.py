import numpy as np
import random
import torch
import torch.multiprocessing as mp

from core.memory import Memory


class SharedMemory(Memory):
    def __init__(self, args):
        super(SharedMemory, self).__init__(args)

        # params for this memory

        # setup
        self.state0s = torch.zeros((self.memory_size, ) + tuple(self.state_shape))
        self.actions = torch.zeros( self.memory_size, self.action_shape)
        self.rewards = torch.zeros( self.memory_size, self.reward_shape)
        self.state1s = torch.zeros((self.memory_size, ) + tuple(self.state_shape))
        self.terminal1s = torch.zeros(self.memory_size)

        self.state0s.share_memory_()
        self.actions.share_memory_()
        self.rewards.share_memory_()
        self.state1s.share_memory_()
        self.terminal1s.share_memory_()

        self.memory_lock = mp.Lock()

    def _feed(self, experience):
        state0, action, reward, state1, terminal1 = experience

        self.state0s[self.pos][:] = torch.FloatTensor(state0)
        self.actions[self.pos][:] = torch.FloatTensor(action)
        self.rewards[self.pos][:] = torch.FloatTensor(reward)
        self.state1s[self.pos][:] = torch.FloatTensor(state1)
        self.terminal1s[self.pos] = terminal1

        self.pos += 1
        if self.pos == self.memory_size:
            self.full = True
            self.pos = 0

    def _sample(self, batch_size):
        upper_bound = self.memory_size if self.full else self.pos
        sampled_indices = torch.LongTensor(np.random.randint(0, upper_bound, size=self.batch_size))
        return [self.states[sampled_indices],
                self.actions[sampled_indices],
                self.rewards[sampled_indices],
                self.next_states[sampled_indices],
                self.terminals[sampled_indices]]

        upper_bound = self.memory_size if self.full else self.pos
        batch_inds = torch.LongTensor(np.random.randint(0, upper_bound, size=batch_size))
        return (self.state0s[batch_inds],
                self.actions[batch_inds],
                self.rewards[batch_inds],
                self.state1s[batch_inds],
                self.terminal1s[batch_inds])

    def feed(self, experience):
        with self.memory_lock:
            self._feed(experience)

    def sample(self, batch_size):
        with self.memory_lock:
            self._sample(batch_size)
