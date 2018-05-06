import torch
import torch.nn as nn
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from utils.helpers import ensure_global_grads


def continuous_actor(process_ind, args,
                     env_prototype,
                     model_prototype,
                     global_memory,
                     global_model):
    print("---------------------------->", process_ind, "actor")
    # env
    env = env_prototype(args.env_params, process_ind, args.num_envs_per_actor)
    # memory
    # model
    cpu_model = model_prototype(args.model_params)
    # sync global model to local
    cpu_model.load_state_dict(global_model.state_dict())

    # params

    # setup
    cpu_model.eval()
    torch.set_grad_enabled(False)

    # main control loop
    # counters
    step = 0
    episode_steps = 0
    episode_reward = 0.
    total_steps = 0
    total_rewards = 0.
    nepisodes = 0
    nepisodes_solved = 0
    # flags
    flag_reset = True   # True when: terminal1 | episode_steps > self.early_stop
    last_state1 = None
    while step < args.agent_params.steps: # TODO: what should be the condition here???
        print(step)
        # sync global model to local
        cpu_model.load_state_dict(global_model.state_dict())    # TODO: check when to update?
        # # deal w/ reset
        # if flag_reset:
        #     # reset episode stats
        #     episode_steps = 0
        #     episode_reward = 0.
        #     # reset game
        #     env.reset()
        #     # flags
        #     flag_reset = False
        # # run a single step
        # # action = cpu_model()

        # update counters & stats
        step += 1

def continuous_learner(process_ind, args,
                       model_prototype,
                       global_memory,
                       global_model):
    board = SummaryWriter(args.log_dir)
    print("---------------------------->", process_ind, "learner")
    # env
    # memory
    # model
    local_device = torch.device('cuda')
    global_device = torch.device('cpu')
    gpu_model = model_prototype(args.model_params).to(local_device)
    # sync global model to local
    gpu_model.load_state_dict(global_model.state_dict())

    # params
    # criteria and optimizer
    actor_optimizer = args.agent_params.optim(gpu_model.actor.parameters())
    critic_optimizer = args.agent_params.optim(gpu_model.critic.parameters())

    # setup
    gpu_model.train()
    torch.set_grad_enabled(True)

    # main control loop
    step = 0
    for step in range(10): # TODO: what should be the condition here???
        batch_size = 8
        input_dims = [1, 1, 100]
        input = torch.randn([batch_size] + input_dims, requires_grad=True)
        output = gpu_model(input.to(local_device))
        # TODO: this part is completely made up for now
        actor_loss = args.agent_params.value_criteria(output[0], torch.ones_like(output[0]))
        critic_loss = args.agent_params.value_criteria(output[1], torch.ones_like(output[1]))
        # print(actor_loss)
        # print(critic_loss)
        actor_optimizer.zero_grad()
        # actor_loss.backward()
        critic_optimizer.zero_grad()
        # critic_loss.backward()
        (actor_loss+critic_loss).backward()
        nn.utils.clip_grad_norm_(gpu_model.parameters(), 100.)

        # sync local grads to global
        ensure_global_grads(gpu_model, global_model, global_device)
        # actor_optimizer.step()    # TODO: local keeps updating its own? then periodcally copy the global model
        # critic_optimizer.step()   # TODO: local keeps updating its own? then periodcally copy the global model

        # logging
        board.add_scalar("learner/test", torch.randn(1), step)

        # update counters & stats
        step += 1


def continuous_evaluator(process_ind, args,
                         env_prototype,
                         model_prototype,
                         global_model):
    print("---------------------------->", process_ind, "evaluator")
    board = SummaryWriter(args.log_dir)
    # env
    env = env_prototype(args.env_params, process_ind)
    # memory
    # model
    cpu_model = model_prototype(args.model_params)
    # sync global model to local
    cpu_model.load_state_dict(global_model.state_dict())

    # params

    # setup
    cpu_model.eval()
    torch.set_grad_enabled(False)

    # main control loop
    # counters
    step = 0
    while step < args.agent_params.steps:
        # logging
        board.add_scalar("evaluator/test", torch.randn(1), step)

        # update counters & stats
        step += 1


def continuous_tester(process_ind, args,
                      env_prototype,
                      model_prototype,
                      global_model):
    print("---------------------------->", process_ind, "tester")
    # env
    env = env_prototype(args.env_params, process_ind)
    # memory
    # model
    cpu_model = model_prototype(args.model_params)
    # sync global model to local
    cpu_model.load_state_dict(global_model.state_dict())

    # params

    # setup
    cpu_model.eval()
    torch.set_grad_enabled(False)

    # main control loop
