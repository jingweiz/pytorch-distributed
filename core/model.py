import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args,
                 norm_val,
                 input_dims,
                 output_dims,
                 action_dims):
        super(Model, self).__init__()

        # model_params
        self.norm_val = norm_val
        # self.model_device = args.model_device
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_dims = action_dims

    def _init_weights(self):
        raise NotImplementedError("not implemented in base calss")

    def _reset(self):   # NOTE: should be called at each child's __init__
        self._init_weights()
        # self.to(self.model_device)

    def forward(self, input):
        raise NotImplementedError("not implemented in base calss")

    def get_action(self, input):
        raise NotImplementedError("not implemented in base calss")
