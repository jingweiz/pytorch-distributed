import torch.multiprocessing as mp

class GlobalLoggers(object):
    def __init__(self):
        self.actor_step = mp.Value('l', 0)
        self.learner_step = mp.Value('l', 0)

class ActorLoggers(object):
    def __init__(self):
        self.total_steps = mp.Value('l', 0)
        self.total_reward = mp.Value('d', 0.)
        self.nepisodes = mp.Value('l', 0)
        self.nepisodes_solved = mp.Value('l', 0)

class LearnerLoggers(object):
    def __init__(self):
        self.actor_loss = mp.Value('d', 0.)
        self.critic_loss = mp.Value('d', 0.)
        self.loss_counter = mp.Value('l', 0)

class EvaluatorLoggers(object):
    def __init__(self):
        self.total_steps = mp.Value('l', 0)
        self.total_reward = mp.Value('d', 0.)
        self.nepisodes = mp.Value('l', 0)
        self.nepisodes_solved = mp.Value('l', 0)
