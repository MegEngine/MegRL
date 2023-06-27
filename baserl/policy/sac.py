from logging import logProcesses
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from baserl.utils.distributions_utils import Categorical, Independent, Normal
from copy import deepcopy
from baserl.data.provider import Batch, ReplayBuffer

import megengine as mge
import megengine.functional as F

from .ddpg import DDPGPolicy
from baserl.exploration import BaseNoise


class SACPolicy(DDPGPolicy):
    """Implementation of Soft Actor-Critic. arXiv:1812.05905.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param float tau: param for soft update of the target network. Default to 0.005.
    :param float gamma: discount factor, in [0, 1]. Default to 0.99.
    :param (float, torch.Tensor, torch.optim.Optimizer) or float alpha: entropy
        regularization coefficient. Default to 0.2.
        If a tuple (target_entropy, log_alpha, alpha_optim) is provided, then
        alpha is automatically tuned.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param BaseNoise exploration_noise: add a noise to action for exploration.
        Default to None. This is useful when solving hard-exploration problem.
    :param bool deterministic_eval: whether to use deterministic action (mean
        of Gaussian policy) instead of stochastic action sampled by the policy.
        Default to True.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action) or empty string for no bounding.
        Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: float = 0.2,
        reward_normalization: bool = False,
        estimation_step: int = 1,
        exploration_noise: Optional[BaseNoise] = None,
        deterministic_eval: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            None, tau, gamma, exploration_noise,
            reward_normalization, estimation_step, **kwargs
        )

        model.actor.action_scaling = self.action_scaling
        model.actor.action_space = self.action_space
        model.actor.eps = np.finfo(np.float32).eps.item()

        self.model = model
        self.model_old = deepcopy(model)
        self.model_old.eval()

        self._is_auto_alpha = False
        self._alpha = alpha

        self._deterministic_eval = deterministic_eval
        self.__eps = np.finfo(np.float32).eps.item()

    def train(self, mode: bool = True) -> "SACPolicy":
        self.training = mode
        self.model.train(mode)
        return self

    def sync_weight(self) -> None:
        self.soft_update(self.model_old.critic1, self.model.critic1, self.tau)
        self.soft_update(self.model_old.critic2, self.model.critic2, self.tau)

    def forward(  # type: ignore
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        input: str = "obs",
        return_type : str = "batch",
        **kwargs: Any,
        ):
        obs = batch[input]
        net_inputs = {
            'obs' : mge.tensor(batch[input]),
            'state' : state,
            'info' : batch.info
        }
        logits, hidden = self.model.actor.network_forward(net_inputs)
        assert isinstance(logits, tuple)
        dist = Independent(Normal(*logits), 1)
        if self._deterministic_eval and not self.training:
            act = logits[0]
        else:
            act = dist.rsample()
        
        log_prob = F.expand_dims(dist.log_prob(act), axis=-1)
        
        if self.action_scaling and self.action_space is not None:
            action_scale = 1.0
        else:
            action_scale = 1.0  # type: ignore
        
        squashed_action = F.tanh(act)
        log_prob = log_prob - F.log(
            action_scale * (1 - F.pow(squashed_action, 2)) + self.__eps
        ).sum(-1, keepdims=True)

        if return_type == 'batch':
            return Batch(
                logits=logits,
                act=squashed_action,
                state=hidden,
                log_prob=log_prob,
                dist=dist,
            )
        elif return_type == 'dict':
            return {
                'logits' : logits,
                'act' : squashed_action,
                'state' : hidden,
                'log_prob' : log_prob,
                "dist": dist,
            }

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray):
        batch = buffer[indices]  # batch.obs: s_{t+n}
        obs_next_result = self(batch, input="obs_next")
        act_ = obs_next_result.act
        c_input = {
            'obs' : mge.Tensor(batch.obs_next),
            'act' : mge.Tensor(act_)
        }
        target_q = F.minimum(
            self.model_old.critic1.network_forward(c_input),
            self.model_old.critic2.network_forward(c_input),
        ) - self._alpha * obs_next_result.log_prob
        return target_q

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # shouldn't reach here
        print("???????????????????")
        return
