import numpy as np
import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp

from utils.options import Options
from utils.loggers import GlobalLoggers, ActorLoggers, LearnerLoggers, EvaluatorLoggers
from utils.factory import LoggerDict, ActorDict, LearnerDict, EvaluatorDict, TesterDict
from utils.factory import EnvDict, MemoryDict, ModelDict

if __name__ == '__main__':
    mp.set_start_method("spawn")

    opt = Options()
    torch.manual_seed(opt.seed)

    env_prototype = EnvDict[opt.env_type]
    memory_prototype = MemoryDict[opt.memory_type]
    model_prototype = ModelDict[opt.model_type]

    # dummy env to get state/action/reward_shape
    dummy_env = env_prototype(opt.env_params, 0)
    opt.state_shape = dummy_env.state_shape
    opt.action_shape = dummy_env.action_shape
    opt.reward_shape = opt.agent_params.num_tasks
    opt.terminal_shape = opt.agent_params.num_tasks
    del dummy_env
    # shared memory
    opt.memory_params.state_shape = opt.state_shape
    opt.memory_params.action_shape = opt.action_shape
    opt.memory_params.reward_shape = opt.reward_shape
    opt.memory_params.terminal_shape = opt.terminal_shape
    global_memory = memory_prototype(opt.memory_params)
    # shared model
    global_model = model_prototype(opt.model_params, opt.state_shape, opt.action_shape)
    global_model.share_memory() # gradients are allocated lazily, so they are not shared here
    # optimizer
    global_actor_optimizer = opt.agent_params.optim(global_model.actor.parameters())
    global_critic_optimizer = opt.agent_params.optim(global_model.critic.parameters())
    global_optimizers = [global_actor_optimizer,
                         global_critic_optimizer]
    # loggers
    global_loggers = GlobalLoggers()
    actor_loggers = ActorLoggers()
    learner_loggers = LearnerLoggers()
    evaluator_loggers = EvaluatorLoggers()

    processes = []
    if opt.mode == 1:
        # logger
        logger_fn = LoggerDict[opt.agent_type]
        p = mp.Process(target=logger_fn,
                       args=(0, opt,
                             global_loggers,
                             actor_loggers,
                             learner_loggers,
                             evaluator_loggers
                            ))
        p.start()
        processes.append(p)
        # actor
        actor_fn = ActorDict[opt.agent_type]
        for process_ind in range(opt.num_actors):
            p = mp.Process(target=actor_fn,
                           args=(process_ind+1, opt,
                                 global_loggers,
                                 actor_loggers,
                                 env_prototype,
                                 model_prototype,
                                 global_memory,
                                 global_model
                                ))
            p.start()
            processes.append(p)
        # learner
        learner_fn = LearnerDict[opt.agent_type]
        for process_ind in range(opt.num_learners):
            p = mp.Process(target=learner_fn,
                           args=(opt.num_actors+process_ind+1, opt,
                                 global_loggers,
                                 learner_loggers,
                                 model_prototype,
                                 global_memory,
                                 global_model,
                                 global_optimizers
                                ))
            p.start()
            processes.append(p)
        # evaluator
        evaluator_fn = EvaluatorDict[opt.agent_type]
        p = mp.Process(target=evaluator_fn,
                       args=(opt.num_actors+opt.num_learners+1, opt,
                             global_loggers,
                             evaluator_loggers,
                             env_prototype,
                             model_prototype,
                             global_model
                            ))
        p.start()
        processes.append(p)
    elif opt.mode == 2:
        # tester
        tester_fn = TesterDict[opt.agent_type]
        p = mp.Process(target=evaluator_fn,
                       args=(opt.num_actors+opt.num_learners+2, opt,
                             global_loggers,
                             env_prototype,
                             model_prototype,
                             global_model))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
