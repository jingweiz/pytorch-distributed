import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import Model


class DdpgMlpModel(Model):
    def __init__(self, args, norm_val, input_dims, output_dims, action_dims):
        super(DdpgMlpModel, self).__init__(args, norm_val, input_dims, output_dims, action_dims)

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

    def forward_actor(self, input):
        input = input.view(input.size(0), -1)
        action = self.actor(input)
        return action

    def forward_critic(self, input, action):
        input = input.view(input.size(0), -1)
        qx = self.critic[0](input)
        qvalue = self.critic[1](torch.cat((qx, action), 1))
        return qvalue

    def forward(self, input):
        action = self.forward_actor(input)
        qvalue = self.forward_critic(input, action)
        return action, qvalue

    def get_action(self, input, noise=0., device=torch.device('cpu')):
        input = torch.FloatTensor(input).unsqueeze(0).to(device)
        action = self.forward_actor(input)
        action = np.array([[action.item()]])
        return action + noise, 0., 0. # TODO: enable_per
