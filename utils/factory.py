from core.single_processes.loggers import ddpg_logger
from core.single_processes.actors import ddpg_actor
from core.single_processes.learners import ddpg_learner
from core.single_processes.evaluators import evaluator
from core.single_processes.testers import tester
LoggersDict = {"discrete":   None,
               "continuous": ddpg_logger}
ActorsDict = {"discrete":   None,
              "continuous": ddpg_actor}
LearnersDict = {"discrete":   None,
                "continuous": ddpg_learner}
EvaluatorsDict = {"discrete":   evaluator,
                  "continuous": evaluator}
TestersDict = {"discrete":   tester,
               "continuous": tester}

from core.envs.gym_env import GymEnv
EnvsDict = {"gym": GymEnv}  # gym wrapper

from core.memories.shared_memory import SharedMemory
MemoriesDict = {"shared": SharedMemory,
                "none":   None}

from core.models.discrete_mlp_model import DiscreteMlpModel
from core.models.continuous_mlp_model import ContinuousMlpModel
ModelsDict = {"discrete-mlp":   DiscreteMlpModel,
              "continuous-mlp": ContinuousMlpModel}
