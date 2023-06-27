import time


from .offpolicy_trainer import OffPolicyTrainer


class OffPolicyDDPGTrainer(OffPolicyTrainer):
    """
    Attributes:
        progress: training process. Contains basic informat such as current iter, max iter.
        model: trained model.
        solver: solver that contains optimizer, grad_manager and so on.
        dataloader: data provider.
        meter: meters to log, such as train_time, losses.
    """

    def __init__(self, cfg, *args, **kwargs):
        """
        Args:
            cfg (Config): config which describes training process.
        """
        super().__init__(cfg, *args, **kwargs)
 

    def model_step(self, model_inputs):
        """
        :meth:`model_step` should be called by :meth:`train_one_iter`, it defines
        basic logic of updating model's parameters.

        Args:
            model_inputs: input of models.
        """
        solver_critic = self.solver["solver_critic"]
        model_critic = getattr(self.model, "critic")
        outputs_critic = solver_critic.minimize(model_critic, model_inputs)

        solver_actor = self.solver["solver_actor"]
        model_actor = getattr(self.model, "actor")
        model_inputs["critic_net"] = model_critic
        outputs_actor = solver_actor.minimize(model_actor, model_inputs)
        
        if self.enable_ema:
            self.ema.step()

        model_outputs = {
            "critic_loss" : outputs_critic["critic_loss"],
            "actor_loss" : outputs_actor["actor_loss"]
        }

        self.policy.sync_weight()
        return model_outputs
