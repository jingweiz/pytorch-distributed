import torch
import torch.nn as nn
import torch.multiprocessing as mp

from utils.options import Options

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

def actor_process(process_ind, args, global_model):
    print("    actor_process --->", process_ind)
    cpu_model = MyModel(args.model_params)
    cpu_model.load_state_dict(global_model.state_dict())
    print("    actor_process ---> cpu_model", cpu_model.actor[0].weight.device)

def learner_process(process_ind, args, global_model):
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

def evaluator_process(process_ind, args, global_model):
    print("evaluator_process --->", process_ind)
    cpu_model = MyModel(args.model_params)
    cpu_model.load_state_dict(global_model.state_dict())
    print("evaluator_process ---> cpu_model", cpu_model.actor[0].weight.device)

if __name__ == '__main__':
    mp.set_start_method("spawn")

    opt = Options()
    torch.manual_seed(opt.seed)

    global_model = MyModel(opt.model_params)
    global_model.share_memory() # gradients are allocated lazily, so they are not shared here

    processes = []
    for process_ind in range(opt.num_actors):
        p = mp.Process(target=actor_process, args=(process_ind, opt, global_model))
        p.start()
        processes.append(p)
    for process_ind in range(opt.num_learners):
        p = mp.Process(target=learner_process, args=(opt.num_actors+process_ind, opt, global_model))
        p.start()
        processes.append(p)
    p = mp.Process(target=evaluator_process, args=(opt.num_actors+opt.num_learners, opt, global_model))
    p.start()
    processes.append(p)


    for p in processes:
        p.join()
