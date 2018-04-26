import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        # model_params
        # self.model_device = args.model_device

    def _init_weights(self):
        raise NotImplementedError("not implemented in base calss")

    def _reset(self):   # NOTE: should be called at each child's __init__
        self._init_weights()
        # self.to(self.model_device)

    def forward(self, input):
        raise NotImplementedError("not implemented in base calss")
