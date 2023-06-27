from baserl.data.provider import batch
import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np

from baserl import layers
from .base_net import BaseNet
from .mlp import MLP
from .cnn import CNN

from baserl.layers import safelog

class DQNNet(BaseNet):
    def __init__(self, model_type="MLP", model_config={}):
        super().__init__()
        self.model = eval(model_type)(**model_config) # compatiable to CNN/MLP

    def network_forward(self, inputs):
        logits, state = self.model(inputs['obs'])
        state = inputs['state']
        return logits, state

    def get_losses(self, inputs):
        q, state = self.network_forward(inputs)
        q = q[np.arange(len(q)), inputs['batch'].act]

        weights = inputs['batch'].pop('weights', 1.0)
        returns = mge.tensor(inputs['batch'].returns).flatten()
        td_error = returns - q
        loss = (F.pow(td_error, 2) * weights).mean()
        inputs['batch'].weight = td_error

        return {
            'td_loss' : loss,
        }

    def inference(self, inputs):
        logits = self.model(inputs['obs'])
        results = F.argmax(logits, axis=1).astype("uint8")
        return results