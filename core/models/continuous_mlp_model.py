import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import Model


class ContinuousMlpModel(Model):
    def __init__(self, args, input_dims, output_dims):
        super(ContinuousMlpModel, self).__init__(args, input_dims, output_dims)

        # model_params for this model

        # actor
        self.actor = nn.Sequential(
            nn.Linear(self.input_dims[0] * self.input_dims[1] * self.input_dims[2], 300),
            nn.Tanh(),
            nn.Linear(300, 200),
            nn.Tanh(),
            nn.Linear(200, self.output_dims),
            nn.Tanh(),
        )

        # critic
        self.critic = nn.ModuleList()
        self.critic.append(nn.Sequential(
            nn.Linear(self.input_dims[0] * self.input_dims[1] * self.input_dims[2], 400),
            nn.Tanh()
        ))
        self.critic.append(nn.Sequential(
            nn.Linear(400 + self.output_dims, 300),
            nn.Tanh(),
            nn.Linear(300, 1)
        ))

        # reset
        self._reset()

    def _init_weights(self):
        bound = 3e-3

        # actor
        nn.init.xavier_uniform_(self.actor[0].weight.data)
        nn.init.constant_(self.actor[0].bias.data, 0)
        nn.init.xavier_uniform_(self.actor[2].weight.data)
        nn.init.constant_(self.actor[2].bias.data, 0)
        nn.init.uniform_(self.actor[4].weight.data, -bound, bound)
        nn.init.constant_(self.actor[4].bias.data, 0)
        # critic
        nn.init.xavier_uniform_(self.critic[0][0].weight.data)
        nn.init.constant_(self.critic[0][0].bias.data, 0)
        nn.init.xavier_uniform_(self.critic[1][0].weight.data)
        nn.init.constant_(self.critic[1][0].bias.data, 0)
        nn.init.uniform_(self.critic[1][2].weight.data, -bound, bound)
        nn.init.constant_(self.critic[1][2].bias.data, 0)

    def forward_actor(self, input_vb):
        input_vb = input_vb.view(input_vb.size(0), -1)
        action_vb = self.actor(input_vb)
        return action_vb

    def forward_critic(self, input_vb, action_vb):
        input_vb = input_vb.view(input_vb.size(0), -1)
        qx_vb = self.critic[0](input_vb)
        qvalue_vb = self.critic[1](torch.cat((qx_vb, action_vb), 1))
        return qvalue_vb

    def forward(self, input_vb):
        action_vb = self.forward_actor(input_vb)
        qvalue_vb = self.forward_critic(input_vb, action_vb)
        return action_vb, qvalue_vb
