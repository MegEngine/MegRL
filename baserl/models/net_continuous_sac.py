from logging import logProcesses
from termios import CR1
from cv2 import circle
from baserl.models.base_net import BaseNet
import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np



from baserl import layers
from baserl.layers import safelog, linear_init
from .net_continous import ActorProb, Critic

from baserl.utils.extra_utils import categorial_entropy, logits_to_probs,normal_rsample, normal_log_prob, independent_log_prob

class ActorContSAC(ActorProb):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_losses(self, inputs):
        batch = inputs['batch']

        policy = inputs['policy']
        obs_result = policy(batch, return_type='dict')

        act = obs_result['act']
        log_prob = obs_result['log_prob']

        critic1 = inputs['critic1_net']
        critic2 = inputs['critic2_net']

        c_input = {'obs' : mge.Tensor(batch.obs),
            'act' : act}

        current_q1a = critic1.network_forward(c_input).flatten()
        current_q2a = critic2.network_forward(c_input).flatten()
        actor_loss = (self._alpha * log_prob.flatten() - F.minimum(current_q1a, current_q2a)).mean()
        

        return {'actor_loss':actor_loss}

class CriticContSAC(Critic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_losses(self, inputs):
        batch = inputs['batch']
        net_inputs = {
            'obs': mge.Tensor(batch.obs),
            'act': mge.Tensor(batch.act)
        }
        weight = getattr(batch, "weight", 1.0)
        current_q = self.network_forward(net_inputs).flatten()
        target_q = batch.returns.flatten()
        td = current_q - target_q
        critic_loss = (F.pow(td, 2) * weight).mean()

        return {'critic_loss' : critic_loss}

class ActorCriticContSAC(BaseNet):
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