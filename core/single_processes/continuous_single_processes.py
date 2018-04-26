import torch
import torch.nn as nn
import torch.multiprocessing as mp

from utils.helpers import ensure_global_grads


def continuous_actor(process_ind, args,
                     model_prototype,
                     global_model):
    # init
    print("    actor_process --->", process_ind)
    cpu_model = model_prototype(args.model_params)
    # sync global model to local
    cpu_model.load_state_dict(global_model.state_dict())

    # params

    # setup
    cpu_model.eval()
    torch.set_grad_enabled(False)

    # act
    for i in range(10):
        # sync global model to local
        cpu_model.load_state_dict(global_model.state_dict())


def continuous_learner(process_ind, args,
                       model_prototype,
                       global_model):
    # init
    print("  learner_process --->", process_ind)
    local_device = torch.device('cuda')
    global_device = torch.device('cpu')
    gpu_model = model_prototype(args.model_params).to(local_device)
    # sync global model to local
    gpu_model.load_state_dict(global_model.state_dict())

    # params
    # optimizer
    actor_optimizer = args.agent_params.optim(gpu_model.actor.parameters())
    critic_optimizer = args.agent_params.optim(gpu_model.critic.parameters())

    # setup
    gpu_model.train()
    torch.set_grad_enabled(True)

    # learn
    step = 0
    for step in range(10):
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


def continuous_evaluator(process_ind, args,
                         model_prototype,
                         global_model):
    # init
    print("evaluator_process --->", process_ind)
    cpu_model = model_prototype(args.model_params)
    # sync global model to local
    cpu_model.load_state_dict(global_model.state_dict())

    # params

    # setup
    cpu_model.eval()
    torch.set_grad_enabled(False)

    # eval


def continuous_tester(process_ind, args,
                      model_prototype,
                      global_model):
    # init
    print("   tester_process --->", process_ind)
    cpu_model = model_prototype(args.model_params)
    # sync global model to local
    cpu_model.load_state_dict(global_model.state_dict())

    # params

    # setup
    cpu_model.eval()
    torch.set_grad_enabled(False)

    # test
