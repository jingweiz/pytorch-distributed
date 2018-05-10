import torch
import torch.nn as nn
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from utils.helpers import Experience, reset_experience
from utils.helpers import ensure_global_grads


def continuous_logger(process_ind, args,
                      loggers):
    print("---------------------------->", process_ind, "logger")
    # loggers
    global_actor_step, global_learner_step = loggers

    # set up board
    board = SummaryWriter(args.log_dir)
    # board.add_text('config', str(args.num_actors) + 'actors(x ' +
    #                          str(args.num_envs_per_actor) + 'envs) + ' +
    #                          str(args.num_learners) + 'learners' + ' | ' +
    #                          args.agent_type + ' | ' +
    #                          args.env_type + ' | ' + args.game + ' | ' +
    #                          args.memory_type + ' | ' +
    #                          args.model_type)
    # for i in range(100):
    #     board.add_scalar("stats/test", torch.randn(1), i)
    # while global_learner_step.value < args.agent_params.steps:
    #     print("logger ---> global_actor_step   --->", global_actor_step.value)
    #     print("logger ---> global_learner_step --->", global_learner_step.value)


def continuous_actor(process_ind, args,
                     loggers,
                     env_prototype,
                     model_prototype,
                     global_memory,
                     global_model):
    print("---------------------------->", process_ind, "actor")
    # loggers
    global_actor_step, global_learner_step = loggers
    # env
    env = env_prototype(args.env_params, process_ind, args.num_envs_per_actor)
    # memory
    # model
    local_device = torch.device('cpu')
    local_model = model_prototype(args.model_params, args.state_shape, args.action_shape).to(local_device)
    # sync global model to local
    local_model.load_state_dict(global_model.state_dict())

    # params

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
    last_state1 = None
    while global_learner_step.value < args.agent_params.steps:
        # deal w/ reset
        if flag_reset:
            # sync global model to local before every new episode # TODO: check when to update?
            local_model.load_state_dict(global_model.state_dict())
            # reset episode stats
            episode_steps = 0
            episode_reward = 0.
            # reset game
            experience = env.reset()
            assert experience.state1 is not None
            last_state1 = experience.state1
            # flags
            flag_reset = False

        # run a single step
        action = local_model.get_action(experience.state1) # TODO: add noise???
        experience = env.step(action)

        # push to memory
        global_memory.feed((last_state1,
                            experience.action,
                            experience.reward,
                            experience.state1,
                            experience.terminal1))
        last_state1 = experience.state1

        # check conditions & update flags
        if experience.terminal1:
            nepisodes_solved += 1
            flag_reset = True
        if args.env_params.early_stop and (episode_steps + 1) >= args.env_params.early_stop:
            flag_reset = True

        # update counters & stats
        with global_actor_step.get_lock():
            global_actor_step.value += 1
        step += 1
        episode_steps += 1
        episode_reward += experience.reward
        if flag_reset:
            nepisodes += 1
            total_steps + episode_steps
            total_reward += episode_reward
        # print("  actor --->   global_actor_step --->", global_actor_step.value, step, global_memory.size)

        # report training stats

        # evaluation & checkpointing


def continuous_learner(process_ind, args,
                       loggers,
                       model_prototype,
                       global_memory,
                       global_model):
    print("---------------------------->", process_ind, "learner")
    # loggers
    global_actor_step, global_learner_step = loggers
    # env
    # memory
    # model
    local_device = torch.device('cuda')
    global_device = torch.device('cpu')
    local_model = model_prototype(args.model_params, args.state_shape, args.action_shape).to(local_device)
    # sync global model to local
    local_model.load_state_dict(global_model.state_dict())

    # params
    # criteria and optimizer
    actor_optimizer = args.agent_params.optim(local_model.actor.parameters())
    critic_optimizer = args.agent_params.optim(local_model.critic.parameters())

    # setup
    local_model.train()
    torch.set_grad_enabled(True)

    # main control loop
    step = 0
    while global_learner_step.value < args.agent_params.steps:# and global_actor_step.value >= args.agent_params.learn_start:#TODO: should use memory.size to judge, but that value is misfunctioning !!!
        # input = global_memory.sample(args.agent_params.batch_size)
        input = torch.randn([args.agent_params.batch_size] + args.state_shape, requires_grad=True)
        print("input.size() --->", input.size())
        output = local_model(input.to(local_device))
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
        nn.utils.clip_grad_norm_(local_model.parameters(), 100.)

        # sync local grads to global
        ensure_global_grads(local_model, global_model, global_device)
        # actor_optimizer.step()    # TODO: local keeps updating its own? then periodcally copy the global model
        # critic_optimizer.step()   # TODO: local keeps updating its own? then periodcally copy the global model

        # update counters & stats
        with global_learner_step.get_lock():
            global_learner_step.value += 1
        step += 1
        print("learner ---> global_learner_step --->", global_learner_step.value, global_memory.size, global_memory.full.value)


def continuous_evaluator(process_ind, args,
                         loggers,
                         env_prototype,
                         model_prototype,
                         global_model):
    print("---------------------------->", process_ind, "evaluator")
    # env
    env = env_prototype(args.env_params, process_ind)
    # memory
    # model
    local_device = torch.device('cpu')
    local_model = model_prototype(args.model_params, args.state_shape, args.action_shape).to(local_device)
    # sync global model to local
    local_model.load_state_dict(global_model.state_dict())

    # params

    # setup
    local_model.eval()
    torch.set_grad_enabled(False)

    # main control loop
    # counters
    step = 0
    while step < args.agent_params.steps:
        # update counters & stats
        step += 1


def continuous_tester(process_ind, args,
                      loggers,
                      env_prototype,
                      model_prototype,
                      global_model):
    print("---------------------------->", process_ind, "tester")
    # env
    env = env_prototype(args.env_params, process_ind)
    # memory
    # model
    local_device = torch.device('cpu')
    local_model = model_prototype(args.model_params, args.state_shape, args.action_shape).to(local_device)
    # sync global model to local
    local_model.load_state_dict(global_model.state_dict())

    # params

    # setup
    local_model.eval()
    torch.set_grad_enabled(False)

    # main control loop
