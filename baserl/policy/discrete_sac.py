from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from baserl.utils.distributions_utils import Categorical

from baserl.data.provider import Batch, ReplayBuffer
from .sac import SACPolicy

import megengine as mge
import megengine.functional as F


class DiscreteSACPolicy(SACPolicy):
    """Implementation of SAC for Discrete Action Settings. arXiv:1910.07207.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s -> Q(s))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s -> Q(s))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param float tau: param for soft update of the target network. Default to 0.005.
    :param float gamma: discount factor, in [0, 1]. Default to 0.99.
    :param (float, torch.Tensor, torch.optim.Optimizer) or float alpha: entropy
        regularization coefficient. Default to 0.2.
        If a tuple (target_entropy, log_alpha, alpha_optim) is provided, the
        alpha is automatically tuned.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.

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
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model,
            tau,
            gamma,
            alpha,
            reward_normalization,
            estimation_step,
            action_scaling=False,
            action_bound_method="",
            **kwargs
        )
        self._alpha: Union[float, mge.Tensor]

    def forward(  # type: ignore
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        net_inputs = {
            'obs' : mge.tensor(batch[input]),
            'state' : state,
            'info' : batch.info
        }
        logits, hidden = self.model.actor.network_forward(net_inputs)

        logits = F.softmax(logits, axis=-1) # 0611 logits->probs
        dist = Categorical(logits)
        if self._deterministic_eval and not self.training:
            act = F.argmax(logits, axis=-1)
        else:
            act = dist.sample()
        act_mge = mge.Tensor(act.numpy())
        return Batch(logits=logits, act=act_mge, state=hidden, dist=dist)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray):
        batch = buffer[indices]  # batch.obs: s_{t+n}
        obs_next_result = self(batch, input="obs_next")
        dist = obs_next_result.dist
        target_q = dist.probs * F.minimum(
            self.model_old.critic1.network_forward(
                {'obs': mge.Tensor(batch.obs_next)}),
            self.model_old.critic2.network_forward(
                {'obs': mge.Tensor(batch.obs_next)}),
        )

        target_q = target_q.sum(axis=-1) + self._alpha * dist.entropy()
        return target_q

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        return super().learn(batch, **kwargs)

    def exploration_noise(self, act: Union[np.ndarray, Batch],
                          batch: Batch) -> Union[np.ndarray, Batch]:
        return act