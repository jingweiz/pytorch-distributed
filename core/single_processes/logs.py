import torch.multiprocessing as mp

class GlobalLogs(object):
    def __init__(self):
        self.actor_step = mp.Value('l', 0)
        self.learner_step = mp.Value('l', 0)

class ActorLogs(object):
    def __init__(self):
        self.total_steps = mp.Value('l', 0)
        self.total_reward = mp.Value('d', 0.)
        self.nepisodes = mp.Value('l', 0)
        self.nepisodes_solved = mp.Value('l', 0)

class DQNLearnerLogs(object):
    def __init__(self):
        self.critic_loss = mp.Value('d', 0.)
        self.loss_counter = mp.Value('l', 0)

class DDPGLearnerLogs(object):
    def __init__(self):
        self.actor_loss = mp.Value('d', 0.)
        self.critic_loss = mp.Value('d', 0.)
        self.loss_counter = mp.Value('l', 0)

class EvaluatorLogs(object):
    def __init__(self):
        self.total_steps = mp.Value('l', 0)
        self.total_reward = mp.Value('d', 0.)
        self.nepisodes = mp.Value('l', 0)
        self.nepisodes_solved = mp.Value('l', 0)
        self.logger_lock = mp.Value('b', False)
