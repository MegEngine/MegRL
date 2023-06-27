from .base_net import BaseNet
from .mlp import MLP
from .cnn import CNN
from .net_pg import ActorProbContPG
from .net_dqn import DQNNet
from .net_ddpg import ActorDDPG, CriticDDPG, ActorCriticDDPG
from .net_a2c import ActorDisA2C, CriticDisA2C, ActorProbContA2C, CriticContA2C, ActorCriticA2C, ActorCriticA2CCont
from .net_ppo import ActorDisPPO, CriticDisPPO, ActorProbContPPO, CriticContPPO, ActorCriticPPO, ActorCriticPPOCont # FIXME: 0608
from  .net_discrete_sac import ActorSAC, CriticSAC, ActorCriticSAC
from  .net_continuous_sac import ActorContSAC, CriticContSAC, ActorCriticContSAC