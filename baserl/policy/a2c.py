from typing import Any, Dict, List, Optional, Type, ValuesView, Union

import numpy as np

import megengine as mge
import megengine.functional as F

from baserl.data.provider import Batch, ReplayBuffer
from baserl.policy import PGPolicy
from baserl.models.net_a2c import ActorCriticA2C


class A2CPolicy(PGPolicy):
    """Implementation of Synchronous Advantage Actor-Critic. arXiv:1602.01783.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the critic network. (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param float vf_coef: weight for value loss. Default to 0.5.
    :param float ent_coef: weight for entropy loss. Default to 0.01.
    :param float max_grad_norm: clipping gradients in back propagation. Default to
        None.
    :param float gae_lambda: in [0, 1], param for Generalized Advantage Estimation.
        Default to 0.95.
    :param bool reward_normalization: normalize estimated values to have std close to
        1. Default to False.
    :param int max_batchsize: the maximum size of the batch when computing GAE,
        depends on the size of available memory and the memory cost of the
        model; should be as large as possible within the memory constraint.
        Default to 256.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    :param bool deterministic_eval: whether to use deterministic action instead of
        stochastic action sampled by the policy. Default to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor,
        critic,
        dist_fn,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        gae_lambda: float = 0.95,
        max_batchsize: int = 256,
        **kwargs: Any
    ) -> None:
        super().__init__(actor, dist_fn, **kwargs)
        self.critic = critic
        assert 0.0 <= gae_lambda <= 1.0, "GAE lambda should be in [0, 1]."
        self._lambda = gae_lambda
        self._weight_vf = vf_coef
        self._weight_ent = ent_coef
        self._batch = max_batchsize

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        batch = self._compute_returns(batch, buffer, indices)
        return batch

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 4 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``dist`` the action distribution.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        logits, hidden = self.actor.network_forward({
            "obs": mge.tensor(batch.obs),
            "state": state, 
        })
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        if self._deterministic_eval and not self.training:
            if self.action_type == "discrete":
                act = F.argmax(logits, axis=-1)
            elif self.action_type == "continuous":
                act = logits[0]
        else:
            act = dist.sample()
        return Batch(logits=logits, act=act, state=hidden, dist=dist)
    
    def _compute_returns(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        v_s, v_s_ = [], []
        for minibatch in batch.split(self._batch, shuffle=False, merge_last=True):
            v_s.append(self.critic.network_forward(
                {'obs': mge.Tensor(minibatch.obs)}))
            v_s_.append(self.critic.network_forward(
                {'obs': mge.Tensor(minibatch.obs_next)}))

        v_s = F.concat(v_s, axis=0).flatten().numpy()
        batch.v_s = v_s
        v_s_ = F.concat(v_s_, axis=0).flatten().numpy()
        # when normalizing values, we do not minus self.ret_rms.mean to be numerically
        # consistent with OPENAI baselines' value normalization pipeline. Emperical
        # study also shows that "minus mean" will harm performances a tiny little bit
        # due to unknown reasons (on Mujoco envs, not confident, though).
        if self._rew_norm:  # unnormalize v_s & v_s_
            v_s = v_s * np.sqrt(self.ret_rms.var + self._eps)
            v_s_ = v_s_ * np.sqrt(self.ret_rms.var + self._eps)
        unnormalized_returns, advantages = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            v_s_,
            v_s,
            gamma=self._gamma,
            gae_lambda=self._lambda
        )
        if self._rew_norm:
            batch.returns = unnormalized_returns / \
                np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns

        batch.adv = advantages
        return batch

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        return super().learn(batch, **kwargs)
    
    def get_loss(self, minibatch):
        # calculate loss for actor
        dist = self(minibatch).dist
        act = mge.tensor(minibatch.act)
        log_prob = dist.log_prob(act)
        log_prob = log_prob.reshape(len(minibatch.adv), -1).transpose(1, 0) # (0,1)
        adv = mge.tensor(minibatch.adv)
        actor_loss = -(log_prob * adv).mean()
        # calculate loss for critic
        value = self.critic.network_forward(
            {"obs": mge.tensor(minibatch.obs)}
        ).flatten()
        ret = mge.tensor(minibatch.returns)
        vf_loss = F.nn.square_loss(ret, value)
        # calculate regularization and overall loss
        ent_loss = dist.entropy().mean()
        loss = actor_loss + self._weight_vf * vf_loss \
            - self._weight_ent * ent_loss

        return {
            "total_loss": loss,
            "loss/actor": actor_loss,
            "loss/vf": vf_loss,
            "loss/ent": ent_loss,
        }
