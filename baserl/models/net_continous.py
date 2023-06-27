from logging import logProcesses
import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np
import warnings

from baserl import layers
from baserl.layers import safelog, linear_init
from baserl.models.mlp import MLP
from .base_net import BaseNet


SIGMA_MIN = -20
SIGMA_MAX = 2

class Actor(BaseNet):
    def __init__(self, preprocess_net, output_dim, max_action = 1.0):
        super().__init__()

        self._max = float(max_action)
        self.preprocess = preprocess_net
        input_dim = getattr(preprocess_net, "output_dim")
        self.last = M.Sequential(
            M.Linear(input_dim, output_dim),
        )
        self._init_modules()

    def _init_modules(self):
        for m in self.last.modules():
            if isinstance(m, M.Linear):
                linear_init(m)

    def get_losses(self, inputs):
        print("????????") # shouldn't reach here
        return {}

    def network_forward(self, inputs):
        logits, state = self.preprocess(inputs['obs'])
        logits = self.last(logits)
        logits = self._max * F.tanh(logits) 
        return logits, state

    def inference(self, inputs):
        print("????????") # shouldn't reach here
        results = None
        return results

class Critic(BaseNet):
    def __init__(self, preprocess_net, output_dim = 1):
        super().__init__()

        self.preprocess = preprocess_net
        input_dim = getattr(preprocess_net, "output_dim")
        self.last = M.Sequential(
            M.Linear(input_dim, output_dim),
        )        
        self._init_modules()

    def _init_modules(self):
        for m in self.last.modules():
            if isinstance(m, M.Linear):
                linear_init(m)

    def get_losses(self, inputs):
        print("????????") # houldn't reach here
        return {}

    def network_forward(self, inputs):
        obs = inputs['obs']
        if inputs.get('act', None) is not None:
            obs = F.concat([obs, inputs['act']], axis=1)
        logits, state = self.preprocess(obs)
        logits = self.last(logits)
        return logits

    def inference(self, inputs):
        print("????????") # shouldn't reach here
        results = None
        return results


class ActorCritic(BaseNet):
    def __init__(self, actor, critic,):
        super().__init__()

        self.actor = actor
        self.critic = critic

    def network_forward(self, inputs):
        return None, None

    def get_losses(self, inputs):
        return {}

    def inference(self, inputs):
        return None



class ActorProb(BaseNet):
    def __init__(
        self,
        preprocess_net,
        hidden_sizes,
        output_dim,
        max_action: float = 1.0,
        unbounded: bool = False,
        conditioned_sigma: bool = False,
    ) -> None:
        super().__init__()
        if unbounded and not np.isclose(max_action, 1.0):
            warnings.warn(
                "Note that max_action input will be discarded when unbounded is True."
            )
            max_action = 1.0
        self.preprocess = preprocess_net
        self.output_dim = output_dim
        input_dim = getattr(preprocess_net, "output_dim")
        self.mu = MLP(input_dim, hidden_sizes, output_dim)
        self._c_sigma = conditioned_sigma

        if conditioned_sigma:
            self.sigma = MLP(input_dim, hidden_sizes, output_dim)
        else:
            self.sigma_param = mge.Parameter(F.zeros((output_dim, 1)))

        self._max = max_action
        self._unbounded = unbounded

    def network_forward(self, inputs):
        """Mapping: obs -> logits -> (mu, sigma)."""
        state = inputs['state']
        logits, hidden = self.preprocess(inputs['obs'])
        mu, _ = self.mu(logits)
        if not self._unbounded:
            mu = self._max * F.tanh(mu)
        if self._c_sigma:
            sigma_pre = self.sigma(logits)[0]
            sigma = F.exp(F.clip(sigma_pre, SIGMA_MIN, SIGMA_MAX))
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = F.exp(self.sigma_param.reshape(shape) + F.zeros_like(mu))
        return (mu, sigma), state

    def inference(self, outputs):
        print("????????") # shouldn't reach here
        return {}
    
    def get_losses(self, outputs):
        print("????????") # shouldn't reach here
        return {}