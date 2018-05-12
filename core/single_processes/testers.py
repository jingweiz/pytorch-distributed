import torch

from utils.helpers import reset_experience


def tester(process_ind, args,
           logs,
           env_prototype,
           model_prototype,
           global_model):
    # logs
    print("---------------------------->", process_ind, "tester")
    # env
    env = env_prototype(args.env_params, process_ind)
    # memory
    # model
    local_device = torch.device('cpu')
    local_model = model_prototype(args.model_params, args.state_shape, args.action_shape).to(local_device)
    # sync global model to local
    local_model.load_state_dict(global_model.state_dict())

    # params

    # setup
    local_model.eval()
    torch.set_grad_enabled(False)

    # main control loop
