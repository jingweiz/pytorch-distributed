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
        self.pos = mp.Value('l', 0)
        self.full = mp.Value('b', False)

        self.state0s = torch.zeros((self.memory_size, ) + tuple(self.state_shape))
        self.actions = torch.zeros( self.memory_size, self.action_shape)
        self.rewards = torch.zeros( self.memory_size, self.reward_shape)
        self.gamma1s = torch.zeros( self.memory_size, self.gamma_shape)
        self.state1s = torch.zeros((self.memory_size, ) + tuple(self.state_shape))
        self.terminal1s = torch.zeros(self.memory_size, self.terminal_shape)

        self.state0s.share_memory_()
        self.actions.share_memory_()
        self.rewards.share_memory_()
        self.gamma1s.share_memory_()
        self.state1s.share_memory_()
        self.terminal1s.share_memory_()

        self.memory_lock = mp.Lock()

    @property
    def size(self):
        if self.full.value:
            return self.memory_size
        return self.pos.value

    def _feed(self, experience, priority=0.):
        state0, action, reward, gamma1, state1, terminal1 = experience

        self.state0s[self.pos.value][:] = torch.FloatTensor(state0)
        self.actions[self.pos.value][:] = torch.FloatTensor(action)
        self.rewards[self.pos.value][:] = torch.FloatTensor(reward)
        self.gamma1s[self.pos.value][:] = torch.FloatTensor(gamma1)
        self.state1s[self.pos.value][:] = torch.FloatTensor(state1)
        self.terminal1s[self.pos.value][:] = torch.FloatTensor([terminal1]) # TODO: is this the best way to store it???
        self.pos.value += 1
        if self.pos.value == self.memory_size:
            self.full.value = True
            self.pos.value = 0

    def _sample(self, batch_size):
        upper_bound = self.memory_size if self.full.value else self.pos.value
        batch_inds = torch.LongTensor(np.random.randint(0, upper_bound, size=batch_size))
        return (self.state0s[batch_inds],
                self.actions[batch_inds],
                self.rewards[batch_inds],
                self.gamma1s[batch_inds],
                self.state1s[batch_inds],
                self.terminal1s[batch_inds])

    def feed(self, experience, priority=0.):
        with self.memory_lock:
            return self._feed(experience, priority)

    def sample(self, batch_size):
        with self.memory_lock:
            return self._sample(batch_size)
