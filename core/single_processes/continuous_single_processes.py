import torch.nn as nn
import torch.multiprocessing as mp


class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(10, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.actor(x)


# def continuous_actor(process_ind, args, board, global_model):
def continuous_actor(process_ind, args, global_model):
    print("    actor_process --->", process_ind)
    cpu_model = MyModel(args.model_params)
    cpu_model.load_state_dict(global_model.state_dict())
    print("    actor_process ---> cpu_model", cpu_model.actor[0].weight.device)


# def continuous_learner(process_ind, args, board, global_model):
def continuous_learner(process_ind, args, global_model):
    print("  learner_process --->", process_ind)
    cpu_model = MyModel(args.model_params)
    cpu_model.load_state_dict(global_model.state_dict())
    print("  learner_process ---> cpu_model", cpu_model.actor[0].weight.device)
    gpu_model = MyModel(args.model_params).to('cuda')
    gpu_model.load_state_dict(cpu_model.state_dict())
    print("  learner_process ---> gpu_model", gpu_model.actor[0].weight.device)
    cpu_model.load_state_dict(gpu_model.state_dict())
    print("  learner_process ---> cpu_model", cpu_model.actor[0].weight.device)
    global_model.load_state_dict(cpu_model.state_dict())
    print("  learner_process ---> global_model", global_model.actor[0].weight.device)


# def continuous_evaluator(process_ind, args, board, global_model):
def continuous_evaluator(process_ind, args, global_model):
    print("evaluator_process --->", process_ind)
    cpu_model = MyModel(args.model_params)
    cpu_model.load_state_dict(global_model.state_dict())
    print("evaluator_process ---> cpu_model", cpu_model.actor[0].weight.device)


# def continuous_tester(process_ind, args, board, global_model):
def continuous_tester(process_ind, args, global_model):
    print("   tester_process --->", process_ind)
    cpu_model = MyModel(args.model_params)
    cpu_model.load_state_dict(global_model.state_dict())
    print("   tester_process ---> cpu_model", cpu_model.actor[0].weight.device)
