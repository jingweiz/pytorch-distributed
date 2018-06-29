import time
from tensorboardX import SummaryWriter


def dqn_logger(process_ind, args,
               global_loggers,
               actor_loggers,
               learner_loggers,
               evaluator_loggers):
    # loggers
    print("---------------------------->", process_ind, "logger")
    actor_total_nepisodes = 0

    # set up board
    board = SummaryWriter(args.log_dir)

    # start logging
    last_log_time = time.time()
    current_actor_step = 0
    current_learner_step = 0
    while global_loggers.learner_step.value < args.agent_params.steps:
        time.sleep(5)
        if evaluator_loggers.logger_lock.value:
            current_actor_step = global_loggers.actor_step.value
            current_learner_step = global_loggers.learner_step.value
            with evaluator_loggers.logger_lock.get_lock():
                if evaluator_loggers.nepisodes.value > 0:
                    current_learner_step = global_loggers.learner_step.value
                    board.add_scalar("evaluator/avg_steps", evaluator_loggers.total_steps.value/evaluator_loggers.nepisodes.value, current_learner_step)
                    board.add_scalar("evaluator/avg_reward", evaluator_loggers.total_reward.value/evaluator_loggers.nepisodes.value, current_learner_step)
                    board.add_scalar("evaluator/repisodes_solved", evaluator_loggers.nepisodes_solved.value/evaluator_loggers.nepisodes.value, current_learner_step)
                    board.add_scalar("evaluator/nepisodes", evaluator_loggers.nepisodes.value, current_learner_step)
                evaluator_loggers.logger_lock.value = False
        if time.time() - last_log_time > args.agent_params.logger_freq:
            with actor_loggers.nepisodes.get_lock():
                if actor_loggers.nepisodes.value > 0:
                    current_learner_step = global_loggers.learner_step.value
                    actor_total_nepisodes += actor_loggers.nepisodes.value
                    board.add_scalar("actor/avg_steps", actor_loggers.total_steps.value/actor_loggers.nepisodes.value, current_learner_step)
                    board.add_scalar("actor/avg_reward", actor_loggers.total_reward.value/actor_loggers.nepisodes.value, current_learner_step)
                    board.add_scalar("actor/repisodes_solved", actor_loggers.nepisodes_solved.value/actor_loggers.nepisodes.value, current_learner_step)
                    board.add_scalar("actor/total_nframes", current_actor_step, current_learner_step)
                    # board.add_scalar("actor/total_nepisodes", actor_total_nepisodes, current_learner_step)
                    actor_loggers.total_steps.value = 0
                    actor_loggers.total_reward.value = 0.
                    actor_loggers.nepisodes.value = 0
                    actor_loggers.nepisodes_solved.value = 0
            with learner_loggers.loss_counter.get_lock():
                if learner_loggers.loss_counter.value > 0:
                    current_learner_step = global_loggers.learner_step.value
                    # board.add_scalar("learner/actor_loss", learner_loggers.actor_loss.value/learner_loggers.loss_counter.value, current_learner_step)
                    board.add_scalar("learner/critic_loss", learner_loggers.critic_loss.value/learner_loggers.loss_counter.value, current_learner_step)
                    # learner_loggers.actor_loss.value = 0.
                    learner_loggers.critic_loss.value = 0.
                    learner_loggers.loss_counter.value = 0
            last_log_time = time.time()
