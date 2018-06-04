import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import Model


class DQNMlpModel(Model):
    def __init__(self, args, input_dims, output_dims, action_dims):
        super(DQNMlpModel, self).__init__(args, input_dims, output_dims, action_dims)

        # model_params for this model

        self.hidden_dims = 256#16

        # critic
        self.critic = nn.Sequential(
            nn.Linear(self.input_dims[0] * self.input_dims[1] * self.input_dims[2], self.hidden_dims),
            nn.ReLU(),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Linear(self.hidden_dims, self.output_dims),
        )

        # reset
        self._reset()

    def _init_weights(self):
        # bound = 3e-3
        #
        # # critic
        # nn.init.xavier_uniform_(self.critic[0].weight.data)
        # nn.init.constant_(self.critic[0].bias.data, 0)
        # nn.init.xavier_uniform_(self.critic[2].weight.data)
        # nn.init.constant_(self.critic[2].bias.data, 0)
        # nn.init.uniform_(self.critic[4].weight.data, -bound, bound)
        # nn.init.constant_(self.critic[4].bias.data, 0)
        pass

    def forward(self, input):
        input = input.view(input.size(0), -1)
        qvalue = self.critic(input)
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
            action = action.numpy()
        return action
