from core.single_processes.logs import GlobalLogs
from core.single_processes.logs import ActorLogs
from core.single_processes.logs import DQNLearnerLogs, DDPGLearnerLogs
from core.single_processes.logs import EvaluatorLogs
GlobalLogsDict = {"discrete":   GlobalLogs,
                  "continuous": GlobalLogs}
ActorLogsDict = {"discrete":   ActorLogs,
                 "continuous": ActorLogs}
LearnerLogsDict = {"discrete":   DQNLearnerLogs,
                   "continuous": DDPGLearnerLogs}
EvaluatorLogsDict = {"discrete":   EvaluatorLogs,
                     "continuous": EvaluatorLogs}

from core.single_processes.loggers import dqn_logger, ddpg_logger
from core.single_processes.actors import ddpg_actor
from core.single_processes.learners import ddpg_learner
from core.single_processes.evaluators import evaluator
from core.single_processes.testers import tester
LoggersDict = {"discrete":   dqn_logger,
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

from core.models.dqn_mlp_model import DQNMlpModel
from core.models.ddpg_mlp_model import DDPGMlpModel
ModelsDict = {"dqn-mlp":  DQNMlpModel,
              "ddpg-mlp": DDPGMlpModel}
