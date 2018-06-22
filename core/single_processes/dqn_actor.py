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
    states_nstep     = deque(maxlen=args.agent_params.nstep + 1)
    actions_nstep    = deque(maxlen=args.agent_params.nstep)
    rewards_nstep    = deque(maxlen=args.agent_params.nstep)
    terminal1s_nstep = deque(maxlen=args.agent_params.nstep)
    if args.memory_params.enable_per:
        qvalues_nstep = deque(maxlen=args.agent_params.nstep)       # for calculating the initial priority
        max_qvalues_nstep = deque(maxlen=args.agent_params.nstep)   # for calculating the initial priority
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
        action, qvalue, max_qvalue = local_model.get_action(experience.state1, args.memory_params.enable_per, eps)
        # action, qvalue, max_qvalue = local_model.get_action(experience.state1, args.memory_params.enable_per, 1)#eps)
        experience = env.step(action)

        # local buffers for nstep
        states_nstep.append(experience.state1)
        actions_nstep.append(experience.action)
        rewards_nstep.append(experience.reward)
        # rewards_nstep.append(len(states_nstep))#experience.reward)
        terminal1s_nstep.append(experience.terminal1)
        qvalues_nstep.append(qvalue)
        max_qvalues_nstep.append(max_qvalue)

        # print("------------------->")
        # print("=====>", experience.state1.mean(), len(states_nstep), experience.terminal1, experience.action)
        # for i in range(len(actions_nstep)):
        #     if i == (len(actions_nstep)-1):
        #         print("=====>", i, states_nstep[i+1].mean(), states_nstep[i].mean(), rewards_nstep[i], terminal1s_nstep[i], actions_nstep[i])
        #     else:
        #         print("----->", i, states_nstep[i+1].mean(), states_nstep[i].mean(), rewards_nstep[i], terminal1s_nstep[i], actions_nstep[i])
        # # rr = np.sum([rewards_nstep[i] * np.power(args.agent_params.gamma, len(rewards_nstep)-1-i) for i in range(len(rewards_nstep))])
        # rr = np.sum([rewards_nstep[i] * np.power(args.agent_params.gamma, i) for i in range(len(rewards_nstep))])
        # print("the current reward is --->", rr)
        # time.sleep(3)
        # push to memory
        rewards_between = np.sum([rewards_nstep[i] * np.power(args.agent_params.gamma, i) for i in range(len(rewards_nstep))])
        gamma_sn = np.power(args.agent_params.gamma, len(states_nstep)-1)
        priority = 0.
        if args.memory_params.enable_per:   # then use tderr as the initial priority
            priority = abs(rewards_between + gamma_sn * max_qvalues_nstep[-1] - qvalues_nstep[0]) # TODO: currently still one step off
            # print(priority)
        global_memory.feed((states_nstep[0],
                            actions_nstep[0],
                            [rewards_between],
                            [gamma_sn],
                            states_nstep[-1],
                            terminal1s_nstep[0]),
                            priority)

        # check conditions & update flags
        if experience.terminal1:
            # TODO: if terminal, need to forward the latest state, and push the last tuple into memory
            # TODO: add here the extra feeding step
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
