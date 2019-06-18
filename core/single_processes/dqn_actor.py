import time
import numpy as np
from collections import deque
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
    env.train()
    # memory
    # model
    local_device = torch.device('cuda')#('cpu')
    local_model = model_prototype(args.model_params,
                                  args.norm_val,
                                  args.state_shape,
                                  args.action_space,
                                  args.action_shape).to(local_device)
    # sync global model to local
    local_model.load_state_dict(global_model.state_dict())

    # params
    if args.num_actors <= 1:    # NOTE: should avoid this situation, here just for debugging
        eps = 0.1
    else:                       # as described in top of Pg.6
        eps = args.agent_params.eps ** (1. + (process_ind-1)/(args.num_actors-1) * args.agent_params.eps_alpha)

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
    # local buffers for nstep
    states_nstep     = deque(maxlen=args.agent_params.nstep + 2)
    actions_nstep    = deque(maxlen=args.agent_params.nstep + 1)
    rewards_nstep    = deque(maxlen=args.agent_params.nstep + 1)
    terminal1s_nstep = deque(maxlen=args.agent_params.nstep + 1)
    if args.memory_params.enable_per:
        qvalues_nstep = deque(maxlen=args.agent_params.nstep + 1)       # for calculating the initial priority
        max_qvalues_nstep = deque(maxlen=args.agent_params.nstep + 1)   # for calculating the initial priority
    while global_logs.learner_step.value < args.agent_params.steps:
        # deal w/ reset
        if flag_reset:
            # reset episode stats
            episode_steps = 0
            episode_reward = 0.
            # reset game
            experience = env.reset()
            assert experience.state1 is not None
            # local buffers for nstep
            states_nstep.clear()
            states_nstep.append(experience.state1)
            actions_nstep.clear()
            rewards_nstep.clear()
            terminal1s_nstep.clear()
            if args.memory_params.enable_per:
                qvalues_nstep.clear()
                max_qvalues_nstep.clear()
            # flags
            flag_reset = False

        # run a single step
        action, qvalue, max_qvalue = local_model.get_action(experience.state1, args.memory_params.enable_per, eps, device=local_device)
        experience = env.step(action)

        # local buffers for nstep
        states_nstep.append(experience.state1)
        actions_nstep.append(experience.action)
        rewards_nstep.append(experience.reward)
        terminal1s_nstep.append(experience.terminal1)
        if args.memory_params.enable_per:
            qvalues_nstep.append(qvalue)
            max_qvalues_nstep.append(max_qvalue)

        # push to memory
        # NOTE: now states_nstep[-1] has not yet been passed through the model
        # NOTE: so its qvalue & max_qvalue are not yet available for calculating the tderr for priority
        # NOTE: so here we only push the second most recent tuple [-2] into the memory
        # NOTE: and do an extra forward of [-1] only when the current episode terminates
        # NOTE: then push the most recent tuple into memory
        # read as: from state0, take action0, accumulate rewards_between in n step, arrive at stateN, results in terminalN
        # state0: states_nstep[0]
        # action0: actions_nstep[0]
        # rewards_between: discounted sum over rewards_nstep[0] ~ rewards_nstep[-2]-
        # stateN: states_nstep[-2]
        # terminalN: terminal1s_nstep[-2]
        # qvalue0: qvalues_nstep[0]
        # max_qvalueN: max_qvalues_nstep[-1] # NOTE: this stores the value for states_nstep[-2]
        if len(states_nstep) >= 3:
            rewards_between = np.sum([rewards_nstep[i] * np.power(args.agent_params.gamma, i) for i in range(len(rewards_nstep) - 1)])
            gamma_sn = np.power(args.agent_params.gamma, len(states_nstep) - 2)
            priority = 0.
            if args.memory_params.enable_per:   # then use tderr as the initial priority
                priority = abs(rewards_between + gamma_sn * max_qvalues_nstep[-1] - qvalues_nstep[0])
            global_memory.feed((states_nstep[0],
                                actions_nstep[0],
                                [rewards_between],
                                [gamma_sn],
                                states_nstep[-2],
                                terminal1s_nstep[-2]),
                                priority)

        # check conditions & update flags
        if experience.terminal1:
            nepisodes_solved += 1
            flag_reset = True
        if args.env_params.early_stop and (episode_steps + 1) >= args.env_params.early_stop:
            flag_reset = True

        # NOTE: now we do the extra forward step of the most recent state
        # NOTE: then push the tuple into memory, if the current episode ends
        if flag_reset:
            if args.memory_params.enable_per:
                # do an extra forward step of the most recent state [-1]
                # _, qvalue, max_qvalue = local_model.get_action(states_nstep[-1], args.memory_params.enable_per, eps)
                _, _, max_qvalue = local_model.get_action(states_nstep[-1], args.memory_params.enable_per, eps)
            if len(states_nstep) >= (args.agent_params.nstep + 2):    # (nstep+1) experiences available, use states_nstep[1] as s0
                rewards_between = np.sum([rewards_nstep[i] * np.power(args.agent_params.gamma, i - 1) for i in range(1, len(rewards_nstep))])
                gamma_sn = np.power(args.agent_params.gamma, len(states_nstep) - 2)
                priority = 0.
                if args.memory_params.enable_per:   # then use tderr as the initial priority
                    priority = abs(rewards_between + gamma_sn * max_qvalue - qvalues_nstep[1])
                global_memory.feed((states_nstep[1],
                                    actions_nstep[1],
                                    [rewards_between],
                                    [gamma_sn],
                                    states_nstep[-1],
                                    terminal1s_nstep[-1]),
                                    priority)
            else:                                   # not all available, just use the oldest states_nstep[0] as s0
                rewards_between = np.sum([rewards_nstep[i] * np.power(args.agent_params.gamma, i) for i in range(len(rewards_nstep))])
                gamma_sn = np.power(args.agent_params.gamma, len(states_nstep) - 1)
                priority = 0.
                if args.memory_params.enable_per:   # then use tderr as the initial priority
                    priority = abs(rewards_between + gamma_sn * max_qvalue - qvalues_nstep[0])
                global_memory.feed((states_nstep[0],
                                    actions_nstep[0],
                                    [rewards_between],
                                    [gamma_sn],
                                    states_nstep[-1],
                                    terminal1s_nstep[-1]),
                                    priority)

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

        # sync global model to local
        if step % args.agent_params.actor_sync_freq == 0:
            local_model.load_state_dict(global_model.state_dict())

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
