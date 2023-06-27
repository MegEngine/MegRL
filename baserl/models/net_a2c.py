import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np

from baserl import layers
from baserl.layers import safelog, linear_init
from baserl.utils.extra_utils import categorial_entropy

from .base_net import BaseNet
from .net_discrete import Actor as ActorDisA2C
from .net_discrete import Critic as CriticDisA2C
from .net_continous import ActorProb as ActorProbContA2C
from .net_continous import Critic as CriticContA2C

class ActorCriticA2C(BaseNet):
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


class ActorCriticA2CCont(BaseNet):
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
