from collections import namedtuple
# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
# NOTE: used as the return format for Env(), and as the format to push into replay memory for off-policy methods (DQN)
# NOTE: when return from Env(), state0 is always None
Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')


def ensure_global_grads(local_model, global_model, global_device):
    for local_param, global_param in zip(local_model.parameters(),
                                         global_model.parameters()):
        if global_param.grad is not None:
            return
        global_param._grad = local_param.grad.to(global_device)
