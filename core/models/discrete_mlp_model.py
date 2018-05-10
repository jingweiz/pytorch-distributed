import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import Model


class DiscreteMlpModel(Model):
    def __init__(self, args, input_dims, output_dims):
        super(DiscreteMlpModel, self).__init__(args, input_dims, output_dims)

        # model_params for this model

        # critic
        self.critic = nn.Sequential(
            nn.Linear(self.input_dims[0] * self.input_dims[1] * self.input_dims[2], 400),
            nn.Tanh(),
            nn.Linear(400, 300),
            nn.Tanh(),
            nn.Linear(300, self.output_dims),
        )

        # reset
        self._reset()

    def _init_weights(self):
        bound = 3e-3

        # critic
        nn.init.xavier_uniform_(self.critic[0].weight.data)
        nn.init.constant_(self.critic[0].bias.data, 0)
        nn.init.xavier_uniform_(self.critic[2].weight.data)
        nn.init.constant_(self.critic[2].bias.data, 0)
        nn.init.uniform_(self.critic[4].weight.data, -bound, bound)
        nn.init.constant_(self.critic[4].bias.data, 0)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        qvalue = self.critic(input)
        return qvalue

    def get_action(self, input, eps=1.):
        input = torch.FloatTensor(input).unsqueeze(0)
        action = self.forward(input)
        # TODO: episilon greedy
        return action
