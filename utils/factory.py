from core.single_processes.discrete_single_processes import *
from core.single_processes.continuous_single_processes import *
ActorDict = {"discrete":   None,
             "continuous": continuous_actor}            # d3pg
LearnerDict = {"discrete":   None,
               "continuous": continuous_learner}        # d3pg
EvaluatorDict = {"discrete":   None,
                 "continuous": continuous_evaluator}    # d3pg
TesterDict = {"discrete":   None,
              "continuous": continuous_tester}          # d3pg

# from core.envs.gym_env import GymEnv
EnvDict = {"gym": None}   # gym wrapper

MemoryDict = {"shared": None,
              "none":   None}

from core.models.continuous_mlp_model import ContinuousMlpModel
ModelDict = {"discrete-mlp":   None,
             "continuous-mlp": ContinuousMlpModel}
