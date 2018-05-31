import numpy as np
from collections import deque
from copy import copy
import torch

from utils.helpers import reset_experience


def dqn_actor(process_ind, args,
              global_logs,
              actor_logs,
              env_prototype,
              model_prototype,
              global_memory,
              global_model):
    # logs
    print("---------------------------->", process_ind, "actor")

    # env
    env = env_prototype(args.env_params, process_ind, args.num_envs_per_actor)
    # memory
    # model
    local_device = torch.device('cpu')
    local_model = model_prototype(args.model_params,
                                  args.state_shape,
                                  args.action_space,
                                  args.action_shape).to(local_device)
    # sync global model to local
    local_model.load_state_dict(global_model.state_dict())

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
    # local buffers for hist_len && nstep
    state0_stacked = deque(maxlen=args.agent_params.hist_len)
    state1_stacked = deque(maxlen=args.agent_params.hist_len)
    while global_logs.learner_step.value < args.agent_params.steps:
        # deal w/ reset
        if flag_reset:
            # sync global model to local before every new episode # TODO: check when to update?
            local_model.load_state_dict(global_model.state_dict())
            # reset episode stats
            episode_steps = 0
            episode_reward = 0.
            # reset game
            experience = env.reset()
            assert experience.state1 is not None
            # local buffers for hist_len && nstep
            state0_stacked.clear()
            state1_stacked.clear()
            for i in range(args.agent_params.hist_len):
                state1_stacked.append(experience.state1)
            # flags
            flag_reset = False

        # run a single step
        eps = args.agent_params.eps_end + max(0, (args.agent_params.eps_start - args.agent_params.eps_end) * (args.agent_params.eps_decay - max(0, step - args.agent_params.learn_start)) / args.agent_params.eps_decay)
        action = local_model.get_action(np.array(list(state1_stacked)), eps) # NOTE: first converting to list is faster than directly to array
        experience = env.step(action)

        # special treatments for hist_len && nstep before push to memory
        state0_stacked = copy(state1_stacked)
        state1_stacked.append(experience.state1)

        # push to memory
        global_memory.feed((np.array(list(state0_stacked)),
                            experience.action,
                            experience.reward,
                            np.array(list(state1_stacked)),
                            experience.terminal1))

        # check conditions & update flags
        if experience.terminal1:
            nepisodes_solved += 1
            flag_reset = True
        if args.env_params.early_stop and (episode_steps + 1) >= args.env_params.early_stop:
            flag_reset = True

        # update counters & stats
        with global_logs.actor_step.get_lock():
            global_logs.actor_step.value += 1
        step += 1
        episode_steps += 1
        episode_reward += experience.reward
        if flag_reset:
            nepisodes += 1
            total_steps += episode_steps
            total_reward += episode_reward

        # report stats
        if step % args.agent_params.actor_freq == 0: # then push local stats to logger & reset local
            # push local stats to logger
            with actor_logs.nepisodes.get_lock():
                actor_logs.total_steps.value += total_steps
                actor_logs.total_reward.value += total_reward
                actor_logs.nepisodes.value += nepisodes
                actor_logs.nepisodes_solved.value += nepisodes_solved
            # reset local stats
            total_steps = 0
            total_reward = 0.
            nepisodes = 0
            nepisodes_solved = 0
