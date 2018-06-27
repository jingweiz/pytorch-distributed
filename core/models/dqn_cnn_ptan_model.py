import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import Model


class DQNPtanModel(Model):
    def __init__(self, args, input_dims, output_dims, action_dims):
        super(DQNPtanModel, self).__init__(args, input_dims, output_dims, action_dims)

        # model_params for this model

        # critic
        self.critic = nn.ModuleList()
        self.critic.append(nn.Sequential(
            nn.Conv2d(self.input_dims[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        ))

        _conv_out_size = self._get_conv_out_size(self.input_dims)

        self.critic.append(nn.Sequential(
            nn.Linear(_conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dims),
        ))

        # reset
        #self._reset()

    def _get_conv_out_size(self, input_dims):
        out = self.critic[0](torch.zeros(input_dims).unsqueeze(0))
        return int(np.prod(out.size()))

    def _init_weights(self):
        relu_gain = nn.init.calculate_gain('relu')

        # critic
        nn.init.orthogonal_(self.critic[0][0].weight.data, relu_gain)
        nn.init.constant_(self.critic[0][0].bias.data, 0)
        nn.init.orthogonal_(self.critic[0][2].weight.data, relu_gain)
        nn.init.constant_(self.critic[0][2].bias.data, 0)
        nn.init.orthogonal_(self.critic[0][4].weight.data, relu_gain)
        nn.init.constant_(self.critic[0][4].bias.data, 0)
        nn.init.orthogonal_(self.critic[1][0].weight.data, relu_gain)
        nn.init.constant_(self.critic[1][0].bias.data, 0)
        nn.init.orthogonal_(self.critic[1][2].weight.data, relu_gain)
        nn.init.constant_(self.critic[1][2].bias.data, 0)

    def forward(self, input):
        qvalue = self.critic[1](self.critic[0](input/255.0).view(input.size(0), -1))
        return qvalue

    def get_action(self, input, eps=0.):
        input = torch.FloatTensor(input).unsqueeze(0)
        if eps > 0. and np.random.uniform() < eps: # then we choose a random action
            action = np.random.randint(self.output_dims,
                                       size=(input.size(0),
                                             self.action_dims))
        else:
            qvalue = self.forward(input)
            _, action = qvalue.max(dim=1, keepdim=True)
            action = action.cpu().numpy()
        return action
