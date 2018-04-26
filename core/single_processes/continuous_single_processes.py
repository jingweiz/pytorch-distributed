import torch
import torch.nn as nn
import torch.multiprocessing as mp

from utils.helpers import ensure_global_grads


def continuous_actor(process_ind, args,
                     model_prototype,
                     global_model):
    print("    actor_process --->", process_ind)
    cpu_model = model_prototype(args.model_params)
    cpu_model.load_state_dict(global_model.state_dict())
    print("    actor_process ---> cpu_model", cpu_model.actor[0].weight.device)


def continuous_learner(process_ind, args,
                       model_prototype,
                       global_model):
    local_device = torch.device('cuda')
    global_device = torch.device('cpu')

    # sync global model to local
    print("  learner_process --->", process_ind)
    gpu_model = model_prototype(args.model_params).to(local_device)
    gpu_model.load_state_dict(global_model.state_dict())
    print("  learner_process ---> gpu_model", gpu_model.actor[0].weight.device)

    # params
    # optimizer
    actor_optimizer = args.agent_params.optim(gpu_model.actor.parameters())
    critic_optimizer = args.agent_params.optim(gpu_model.critic.parameters())

    # learn
    gpu_model.train()
    torch.set_grad_enabled(True)

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
    print("evaluator_process --->", process_ind)
    cpu_model = model_prototype(args.model_params)
    cpu_model.load_state_dict(global_model.state_dict())
    print("evaluator_process ---> cpu_model", cpu_model.actor[0].weight.device)

    # eval
    cpu_model.eval()
    torch.set_grad_enabled(False)

    pass


def continuous_tester(process_ind, args,
                      model_prototype,
                      global_model):
    print("   tester_process --->", process_ind)
    cpu_model = model_prototype(args.model_params)
    cpu_model.load_state_dict(global_model.state_dict())
    print("   tester_process ---> cpu_model", cpu_model.actor[0].weight.device)

    # test
    cpu_model.eval()
    torch.set_grad_enabled(False)

    pass
