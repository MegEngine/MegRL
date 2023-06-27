from logging import logProcesses
from termios import CR1
from baserl.models.base_net import BaseNet
import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np

from baserl import layers
from baserl.layers import safelog, linear_init
from .net_discrete import Actor, Critic
from baserl.utils.distributions_utils import Categorical

class ActorSAC(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_losses(self, inputs):
        batch = inputs['batch']
        critic1 = inputs['critic1_net']
        critic2 = inputs['critic2_net']
        
        actor_inputs = {'obs' : mge.Tensor(batch.obs) }
        logits, hidden = self.network_forward(actor_inputs)
        logits = F.softmax(logits, axis=-1) # logits->prob
        dist = Categorical(logits)
        entropy = dist.entropy()
        
        current_q1a = critic1.network_forward(
            {'obs': mge.Tensor(batch.obs)}
        ).detach()
        current_q2a = critic2.network_forward(
            {'obs': mge.Tensor(batch.obs)}
        ).detach()
        q = F.minimum(current_q1a, current_q2a)
        
        actor_loss = -(self._alpha * entropy + (dist.probs * q).sum(axis=-1)).mean()

        return {'actor_loss':actor_loss}

class CriticSAC(Critic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_losses(self, inputs):
        batch = inputs['batch']
        weight = batch.pop("weight", 1.0)
        target_q = batch.returns.flatten()
        act = mge.Tensor(batch.act[:, np.newaxis])

        current_q = self.network_forward({'obs': mge.Tensor(batch.obs)})
        current_q = F.gather(current_q ,1, act).flatten()
        td = current_q - target_q
        critic_loss = (F.pow(td, 2) * weight).mean()

        return {'critic_loss' : critic_loss}

class ActorCriticSAC(BaseNet):
    def __init__(self, actor, critic1, critic2):
        super().__init__()

        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2

    def network_forward(self, inputs):
        return {}
    
    def get_losses(self, inputs):
        return {}

    def inference(self, inputs):
        return {}