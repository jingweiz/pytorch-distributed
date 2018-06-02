import numpy as np
from collections import deque
import time
import torch

from utils.helpers import reset_experience


def evaluator(process_ind, args,
              global_logs,
              evaluator_logs,
              env_prototype,
              model_prototype,
              global_model):
    # logs
    print("---------------------------->", process_ind, "evaluator")
    # env
    env = env_prototype(args.env_params, process_ind)
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

    last_eval_time = time.time()
    while global_logs.learner_step.value < args.agent_params.steps:
        if time.time() - last_eval_time > args.agent_params.evaluator_freq:
            # sync global model to local
            local_model.load_state_dict(global_model.state_dict())

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
            state1_stacked = deque(maxlen=args.agent_params.hist_len)
            while step < args.agent_params.evaluator_steps:
                # deal w/ reset
                if flag_reset:
                    # reset episode stats
                    episode_steps = 0
                    episode_reward = 0.
                    # reset game
                    experience = env.reset()
                    assert experience.state1 is not None
                    # local buffers for hist_len && nstep
                    state1_stacked.clear()
                    for i in range(args.agent_params.hist_len):
                        state1_stacked.append(experience.state1)
                    # flags
                    flag_reset = False

                # run a single step
                action = local_model.get_action(np.array(list(state1_stacked)))
                reward = 0.
                for _ in range(args.agent_params.action_repetition):
                    experience = env.step(action)
                    reward += experience.reward
                    if experience.terminal1:
                        break

                # special treatments for hist_len && nstep
                state1_stacked.append(experience.state1)

                # check conditions & update flags
                if experience.terminal1:
                    nepisodes_solved += 1
                    flag_reset = True
                if args.env_params.early_stop and (episode_steps + 1) >= args.env_params.early_stop:
                    flag_reset = True

                # update counters & stats
                step += 1
                episode_steps += 1
                episode_reward += reward
                if flag_reset:
                    nepisodes += 1
                    total_steps += episode_steps
                    total_reward += episode_reward

            # report stats
            # push local stats to logger
            with evaluator_logs.logger_lock.get_lock():
                evaluator_logs.total_steps.value = total_steps
                evaluator_logs.total_reward.value = total_reward
                evaluator_logs.nepisodes.value = nepisodes
                evaluator_logs.nepisodes_solved.value = nepisodes_solved
                evaluator_logs.logger_lock.value = True

            # save model
            print("Saving model " + args.model_name + " ...")
            torch.save(global_model.state_dict(), args.model_name)
            print("Saved  model " + args.model_name + ".")

            last_eval_time = time.time()
