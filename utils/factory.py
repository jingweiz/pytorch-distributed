from core.single_processes.logs import GlobalLogs
from core.single_processes.logs import ActorLogs
from core.single_processes.logs import DQNLearnerLogs, DDPGLearnerLogs
from core.single_processes.logs import EvaluatorLogs
GlobalLogsDict = {"dqn":  GlobalLogs,
                  "ddpg": GlobalLogs}
ActorLogsDict = {"dqn":  ActorLogs,
                 "ddpg": ActorLogs}
LearnerLogsDict = {"dqn":  DQNLearnerLogs,
                   "ddpg": DDPGLearnerLogs}
EvaluatorLogsDict = {"dqn":  EvaluatorLogs,
                     "ddpg": EvaluatorLogs}

from core.single_processes.dqn_logger import dqn_logger
from core.single_processes.ddpg_logger import ddpg_logger
from core.single_processes.dqn_actor import dqn_actor
from core.single_processes.ddpg_actor import ddpg_actor
from core.single_processes.dqn_learner import dqn_learner
from core.single_processes.ddpg_learner import ddpg_learner
from core.single_processes.evaluators import evaluator
from core.single_processes.testers import tester
LoggersDict = {"dqn":  dqn_logger,
               "ddpg": ddpg_logger}
ActorsDict = {"dqn":  dqn_actor,
              "ddpg": ddpg_actor}
LearnersDict = {"dqn":  dqn_learner,
                "ddpg": ddpg_learner}
EvaluatorsDict = {"dqn":  evaluator,
                  "ddpg": evaluator}
TestersDict = {"dqn":  tester,
               "ddpg": tester}

from core.envs.gym_env import GymEnv
EnvsDict = {"gym": GymEnv}  # gym wrapper

from core.memories.shared_memory import SharedMemory
MemoriesDict = {"shared": SharedMemory,
                "none":   None}

from core.models.dqn_mlp_model import DQNMlpModel
from core.models.dqn_cnn_model import DQNCnnModel
from core.models.ddpg_mlp_model import DDPGMlpModel
ModelsDict = {"dqn-mlp":  DQNMlpModel,
              "dqn-cnn":  DQNCnnModel,
              "ddpg-mlp": DDPGMlpModel}
