import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np

from baserl import layers
from baserl.layers import safelog, linear_init
from baserl.utils.extra_utils import categorial_entropy, categorial_log_prob

from .base_net import BaseNet
from .net_discrete import Actor as ActorDisPPO
from .net_discrete import Critic as CriticDisPPO
from .net_continous import ActorProb as ActorProbContPPO
from .net_continous import Critic as CriticContPPO


class ActorCriticPPO(BaseNet):
    def __init__(self,
        actor,
        critic,
        ):
        super().__init__()

        self.actor = actor
        self.critic = critic

    def network_forward(self, inputs):
        print("??????") # shouldn't reach here
        return

    def get_losses(self, inputs):
        print("??????") # shouldn't reach here
        return

    def inference(self, inputs):
        print("??????") # shouldn't reach here
        return

class ActorCriticPPOCont(BaseNet):
    def __init__(self,
        actor,
        critic,
        ):
        super().__init__()

        self.actor = actor
        self.critic = critic

    def network_forward(self, inputs):
        print("??????") # shouldn't reach here
        return

    def get_losses(self, inputs):
        print("??????") # shouldn't reach here
        return

    def inference(self, inputs):
        print("??????") # shouldn't reach here
        return
