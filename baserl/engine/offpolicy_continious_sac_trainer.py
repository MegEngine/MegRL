from .offpolicy_trainer import OffPolicyTrainer


class OffPolicyContinuousSACTrainer(OffPolicyTrainer):
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
        solver_critic1 = self.solver["solver_critic1"]
        model_critic1 = getattr(self.model, "critic1")
        outputs_critic1 = solver_critic1.minimize(model_critic1, model_inputs)

        solver_critic2 = self.solver["solver_critic2"]
        model_critic2 = getattr(self.model, "critic2")
        outputs_critic2 = solver_critic2.minimize(model_critic2, model_inputs)

        solver_actor = self.solver["solver_actor"]
        model_actor = getattr(self.model, "actor")
        model_inputs["critic1_net"] = model_critic1
        model_inputs["critic2_net"] = model_critic2

        model_inputs['policy'] = self.policy # the only difference with discrete
        outputs_actor = solver_actor.minimize(model_actor, model_inputs)
        
        if self.enable_ema:
            self.ema.step()

        model_outputs = {
            "critic1_loss" : outputs_critic1["critic_loss"],
            "critic2_loss" : outputs_critic2["critic_loss"],
            "actor_loss" : outputs_actor["actor_loss"]
        }

        self.policy.sync_weight()
        return model_outputs
