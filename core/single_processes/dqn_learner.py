import time
import torch
import torch.nn as nn

from utils.helpers import update_target_model
from utils.helpers import ensure_global_grads


def dqn_learner(process_ind, args,
                global_logs,
                learner_logs,
                model_prototype,
                global_memory,
                global_model,
                global_optimizer):
    # logs
    print("---------------------------->", process_ind, "learner")
    # env
    # memory
    # model
    local_device = torch.device('cuda:' + str(args.gpu_ind))
    # global_device = torch.device('cpu')
    # local_model = model_prototype(args.model_params,
    #                               args.state_shape,
    #                               args.action_space,
    #                               args.action_shape).to(local_device)
    local_target_model = model_prototype(args.model_params,
                                         args.norm_val,
                                         args.state_shape,
                                         args.action_space,
                                         args.action_shape).to(local_device)
    # # sync global model to local
    # local_model.load_state_dict(global_model.state_dict())
    # update_target_model(local_model, local_target_model) # do a hard update in the beginning
    update_target_model(global_model, local_target_model) # do a hard update in the beginning
    # optimizer
    local_optimizer = args.agent_params.optim(global_model.parameters(),
                                              lr=args.agent_params.lr,
                                              weight_decay=args.agent_params.weight_decay)

    # params

    # setup
    # local_model.train()
    global_model.train()
    torch.set_grad_enabled(True)

    # main control loop
    step = 0
    while global_logs.learner_step.value < args.agent_params.steps:
        if global_memory.size > args.agent_params.learn_start:
            # # sync global model to local
            # local_model.load_state_dict(global_model.state_dict())
            # sample batch from global_memory
            experiences = global_memory.sample(args.agent_params.batch_size)
            state0s, actions, rewards, gamma1s, state1s, terminal1s = experiences

            # learn on this batch - setup
            # global_optimizer.zero_grad()
            # state0s = state0s.to(local_device)
            # state1s = state1s.to(local_device)
            local_optimizer.zero_grad()
            state0s = state0s.cuda(non_blocking=True).to(local_device)
            state1s = state1s.cuda(non_blocking=True).to(local_device)

            # learn on this batch - critic loss
            if args.agent_params.enable_double: # double q
                # predict_next_qvalues = local_model(state1s)
                predict_next_qvalues = global_model(state1s)
                _, max_next_actions = predict_next_qvalues.max(dim=1, keepdim=True)
                target_qvalues = local_target_model(state1s).gather(1, max_next_actions)
            else:                               # dqn
                target_qvalues, _ = local_target_model(state1s).max(dim=1, keepdim=True)
            target_qvalues = rewards.to(local_device) + gamma1s.to(local_device) * target_qvalues.detach() * (1 - terminal1s.to(local_device))
            # predict_qvalues = local_model(state0s).gather(1, actions.long().to(local_device))
            predict_qvalues = global_model(state0s).gather(1, actions.long().to(local_device))
            critic_loss = args.agent_params.value_criteria(predict_qvalues, target_qvalues)

            # local_model.critic.zero_grad()
            critic_loss.backward()
            # nn.utils.clip_grad_value_(local_model.critic.parameters(), args.agent_params.clip_grad)
            nn.utils.clip_grad_value_(global_model.critic.parameters(), args.agent_params.clip_grad)

            # learn on this batch - sync local grads to global
            # ensure_global_grads(local_model, global_model, local_device, global_device)
            # global_optimizer.step()
            local_optimizer.step()

            # update target_model
            # update_target_model(local_model, local_target_model, args.agent_params.target_model_update, step)
            update_target_model(global_model, local_target_model, args.agent_params.target_model_update, step)

            # update counters
            with global_logs.learner_step.get_lock():
                global_logs.learner_step.value += 1
            step += 1

            # report stats
            if step % args.agent_params.learner_freq == 0: # then push local stats to logger & reset local
                learner_logs.critic_loss.value += critic_loss.item()
                learner_logs.loss_counter.value += 1
        else: # wait for memory_size to be larger than learn_start
            time.sleep(1)
