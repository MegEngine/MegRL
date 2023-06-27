import megengine as mge
import megengine.functional as F
import megengine.module as M
from .net_continous import Actor, Critic, ActorCritic

class ActorDDPG(Actor):
    def __init__(self, *args):
        super().__init__(*args)

    def get_losses(self, inputs):
        critic_net = inputs['critic_net']
        batch = inputs['batch']
        actor_inputs = {'obs' : mge.Tensor(batch.obs) }
        
        actions, hidden = self.network_forward(actor_inputs)
        critic_inputs = {'obs' : mge.Tensor(batch.obs), 'act' : actions}
        actor_loss = - critic_net.network_forward(critic_inputs).mean()
        
        return {
            'actor_loss' : actor_loss,
        } 

class CriticDDPG(Critic):
    def __init__(self, *args):
        super().__init__(*args)
    
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
        inputs['batch'].weight = td

        return {
            'critic_loss' : critic_loss,
        }

ActorCriticDDPG = ActorCritic
