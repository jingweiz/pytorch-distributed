import numpy as np
from collections import deque
import torch

from utils.helpers import reset_experience


def tester(process_ind, args,
           env_prototype,
           model_prototype):
    # logs
    print("---------------------------->", process_ind, "tester")
    # env
    env = env_prototype(args.env_params, process_ind)
    env.eval()
    # memory
    # model
    local_device = torch.device('cpu')
    local_model = model_prototype(args.model_params,
                                  args.norm_val,
                                  args.state_shape,
                                  args.action_space,
                                  args.action_shape).to(local_device)
    # sync global model to local
    local_model.load_state_dict(torch.load(args.model_file))

    # params

    # setup
    local_model.eval()
    torch.set_grad_enabled(False)

    # main control loop
    experience = reset_experience()
    # counters
    step = 0
    episode_steps = 0
    episode_reward = 0.
    total_steps = 0
    total_reward = 0.
    nepisodes = 0
    nepisodes_solved = 0
    # flags
    flag_reset = True   # True when: terminal1 | episode_steps > self.early_stop
    while nepisodes < args.agent_params.tester_nepisodes:
        # deal w/ reset
        if flag_reset:
            # reset episode stats
            episode_steps = 0
            episode_reward = 0.
            # reset game
            experience = env.reset()
            assert experience.state1 is not None
            # flags
            flag_reset = False

        # run a single step
        action, _, _ = local_model.get_action(experience.state1)
        experience = env.step(action)

        # check conditions & update flags
        if experience.terminal1:
            nepisodes_solved += 1
            flag_reset = True
        if args.env_params.early_stop and (episode_steps + 1) >= args.env_params.early_stop:
            flag_reset = True

        # update counters & stats
        step += 1
        episode_steps += 1
        episode_reward += experience.reward
        if flag_reset:
            nepisodes += 1
            total_steps += episode_steps
            total_reward += episode_reward
            print("Testing Episode ", nepisodes)

    # report stats
    print("nepisodes:", nepisodes)
    print("avg_steps:", total_steps/nepisodes)
    print("avg_reward:", total_reward/nepisodes)
    print("nepisodes_solved:", nepisodes_solved)
    print("repisodes_solved:", nepisodes_solved/nepisodes)
