import torch
import torch.nn as nn
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from utils.helpers import Experience, reset_experience
from utils.helpers import update_target_model
from utils.helpers import ensure_global_grads


def continuous_logger(process_ind, args,
                      loggers):
    print("---------------------------->", process_ind, "logger")
    # loggers
    global_actor_step, global_learner_step = loggers

    # set up board
    board = SummaryWriter(args.log_dir)
    board.add_text('config', str(args.num_actors) + 'actors(x ' +
                             str(args.num_envs_per_actor) + 'envs) + ' +
                             str(args.num_learners) + 'learners' + ' | ' +
                             args.agent_type + ' | ' +
                             args.env_type + ' | ' + args.game + ' | ' +
                             args.memory_type + ' | ' +
                             args.model_type)
    while global_learner_step.value < args.agent_params.steps:
        board.add_scalar("test/random", torch.randn(1), global_learner_step.value)
        board.add_scalar("info/stats", global_actor_step.value, global_learner_step.value)


def continuous_actor(process_ind, args,
                     loggers,
                     env_prototype,
                     model_prototype,
                     global_memory,
                     global_model):
    # loggers
    print("---------------------------->", process_ind, "actor")
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
    random_process = args.agent_params.random_process(size=args.action_shape,
        theta=0.15, sigma=0.3, n_steps_annealing=args.memory_params.memory_size*100)

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
        action = local_model.get_action(experience.state1, random_process.sample())
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

        # report stats


def continuous_learner(process_ind, args,
                       loggers,
                       model_prototype,
                       global_memory,
                       global_model,
                       global_optimizers):
    # loggers
    print("---------------------------->", process_ind, "learner")
    global_actor_step, global_learner_step = loggers
    # env
    # memory
    # model
    local_device = torch.device('cuda') # TODO: should assign each learner to a seperate gpu
    global_device = torch.device('cpu')
    local_model = model_prototype(args.model_params, args.state_shape, args.action_shape).to(local_device)
    local_target_model = model_prototype(args.model_params, args.state_shape, args.action_shape).to(local_device)
    # sync global model to local
    local_model.load_state_dict(global_model.state_dict())
    update_target_model(local_model, local_target_model) # do a hard update in the beginning
    # optimizers
    global_actor_optimizer, global_critic_optimizer = global_optimizers

    # params

    # setup
    local_model.train()
    torch.set_grad_enabled(True)

    # main control loop
    step = 0
    while global_learner_step.value < args.agent_params.steps:
        if global_memory.size >= args.agent_params.learn_start:
            # sample batch from global_memory
            experiences = global_memory.sample(args.agent_params.batch_size)
            state0s, actions, rewards, state1s, terminal1s = experiences

            # learn on this batch - setup
            state0s = state0s.to(local_device)

            # learn on this batch - actor loss
            _, qvalues = local_model(state0s)
            actor_loss = - qvalues.mean()

            global_actor_optimizer.zero_grad()
            local_model.actor.zero_grad()
            actor_loss.backward()
            # TODO: check here if we need clipping
            nn.utils.clip_grad_value_(local_model.actor.parameters(), args.agent_params.clip_grad)

            # learn on this batch - critic loss
            _, target_qvalues = local_target_model(state1s.to(local_device))
            target_qvalues = rewards.to(local_device) + args.agent_params.gamma * target_qvalues.detach() * (1 - terminal1s.to(local_device))
            predict_qvalues = local_model.forward_critic(state0s, actions.to(local_device))
            critic_loss = args.agent_params.value_criteria(predict_qvalues, target_qvalues)

            global_critic_optimizer.zero_grad()
            local_model.critic.zero_grad()
            critic_loss.backward()
            # TODO: check here if we need clipping
            nn.utils.clip_grad_value_(local_model.critic.parameters(), args.agent_params.clip_grad)

            # learn on this batch - sync local grads to global
            ensure_global_grads(local_model, global_model, global_device)
            global_actor_optimizer.step()
            global_actor_optimizer.step()

            # update counters & stats
            with global_learner_step.get_lock():
                global_learner_step.value += 1
            step += 1

            # report stats


def continuous_evaluator(process_ind, args,
                         loggers,
                         env_prototype,
                         model_prototype,
                         global_model):
    # loggers
    print("---------------------------->", process_ind, "evaluator")
    global_actor_step, global_learner_step = loggers
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

    if True:    # do a evaluation
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
        while step < args.agent_params.eval_steps:
            # deal w/ reset
            if flag_reset:
                # reset episode stats
                episode_steps = 0
                episode_reward = 0.
                # reset game
                experience = env.reset()
                assert experience.state1 is not None
                # flags
                flag_reset = False

            # run a single step
            action = local_model.get_action(experience.state1)
            experience = env.step(action)

            # check conditions & update flags
            if experience.terminal1:
                nepisodes_solved += 1
                flag_reset = True
            if args.env_params.early_stop and (episode_steps + 1) >= args.env_params.early_stop:
                flag_reset = True

            # update counters & stats
            step += 1
            episode_steps += 1
            episode_reward += experience.reward
            if flag_reset:
                nepisodes += 1
                total_steps + episode_steps
                total_reward += episode_reward

            # report stats


def continuous_tester(process_ind, args,
                      loggers,
                      env_prototype,
                      model_prototype,
                      global_model):
    # loggers
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
