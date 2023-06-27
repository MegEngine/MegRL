from typing import Any, Dict, List, Optional, Type

import numpy as np

import megengine as mge
import megengine.functional as F

from baserl.data.provider import Batch, ReplayBuffer
from baserl.policy import A2CPolicy



class PPOPolicy(A2CPolicy):
    r"""Implementation of Proximal Policy Optimization. arXiv:1707.06347.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the critic network. (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param float eps_clip: :math:`\epsilon` in :math:`L_{CLIP}` in the original
        paper. Default to 0.2.
    :param float dual_clip: a parameter c mentioned in arXiv:1912.09729 Equ. 5,
        where c > 1 is a constant indicating the lower bound.
        Default to 5.0 (set None if you do not want to use it).
    :param bool value_clip: a parameter mentioned in arXiv:1811.02553v3 Sec. 4.1.
        Default to True.
    :param bool advantage_normalization: whether to do per mini-batch advantage
        normalization. Default to True.
    :param bool recompute_advantage: whether to recompute advantage every update
        repeat according to https://arxiv.org/pdf/2006.05990.pdf Sec. 3.5.
        Default to False.
    :param float vf_coef: weight for value loss. Default to 0.5.
    :param float ent_coef: weight for entropy loss. Default to 0.01.
    :param float max_grad_norm: clipping gradients in back propagation. Default to
        None.
    :param float gae_lambda: in [0, 1], param for Generalized Advantage Estimation.
        Default to 0.95.
    :param bool reward_normalization: normalize estimated values to have std close
        to 1, also normalize the advantage to Normal(0, 1). Default to False.
    :param int max_batchsize: the maximum size of the batch when computing GAE,
        depends on the size of available memory and the memory cost of the model;
        should be as large as possible within the memory constraint. Default to 256.
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
        eps_clip: float = 0.2,
        dual_clip: Optional[float] = None,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(actor, critic, dist_fn, **kwargs)
        self._eps_clip = eps_clip
        assert dual_clip is None or dual_clip > 1.0, \
            "Dual-clip PPO parameter should greater than 1.0."
        self._dual_clip = dual_clip
        self._value_clip = value_clip

        self._norm_adv = advantage_normalization
        self._recompute_adv = recompute_advantage


    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        if self._recompute_adv:
            # buffer input `buffer` and `indices` to be used in `learn()`.
            self._buffer, self._indices = buffer, indices
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = mge.tensor(batch.act)
        act = mge.tensor(batch.act)
        batch.logp_old = self(batch).dist.log_prob(act)
        return batch

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        return super().learn(batch, **kwargs)
    
    def get_loss(self, minibatch):
        # calculate loss for actor
        dist = self(minibatch).dist
        if self._norm_adv:
            mean, std = minibatch.adv.mean(), minibatch.adv.std()
            minibatch.adv = (minibatch.adv -
                                mean) / (std + self._eps)  # per-batch norm
        act = mge.tensor(minibatch.act)
        ratio = F.exp(dist.log_prob(act) -
                    minibatch.logp_old)
        ratio = ratio.reshape(ratio.shape[0], -1).transpose(1, 0)

        adv = mge.tensor(minibatch.adv)

        surr1 = ratio * adv
        surr2 = F.clip(ratio, 1.0 - self._eps_clip, 1.0 + self._eps_clip) * adv
        if self._dual_clip:
            clip1 = F.minimum(surr1, surr2)
            clip2 = F.maximum(clip1, self._dual_clip * adv)
            clip_loss = -F.where(adv < 0, clip2, clip1).mean()
        else:
            clip_loss = -F.minimum(surr1, surr2).mean()
        # calculate loss for critic
        value = self.critic.network_forward(
            {"obs": mge.tensor(minibatch.obs)}
        ).flatten()
        ret = mge.tensor(minibatch.returns)
        if self._value_clip:
            v_clip = minibatch.v_s + \
                F.clip((value - minibatch.v_s), -self._eps_clip, self._eps_clip)
            vf1 = F.pow(ret - value, 2)
            vf2 = F.pow(ret - v_clip, 2)
            vf_loss = F.maximum(vf1, vf2).mean()
        else:
            vf_loss = F.pow(ret - value, 2).mean()
        # calculate regularization and overall loss
        ent_loss = dist.entropy().mean()
        loss = clip_loss + self._weight_vf * vf_loss \
            - self._weight_ent * ent_loss
        

        return {
            "total_loss": loss,
            "loss/clip": clip_loss,
            "loss/vf": vf_loss,
            "loss/ent": ent_loss,
        }
