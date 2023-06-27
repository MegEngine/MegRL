from logging import logProcesses
from baserl.models.base_net import BaseNet
import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np

from baserl import layers
from baserl.layers import safelog, linear_init


class Actor(BaseNet):
    def __init__(self, 
        preprocess_net,
        output_dim,
        softmax_output=True):

        super().__init__()

        self.preprocess = preprocess_net
        input_dim = getattr(preprocess_net, "output_dim")
        m_list = [M.Linear(input_dim, output_dim)]
        if softmax_output:
            m_list.append(M.Softmax(axis=1))
        self.last = M.Sequential(*m_list)
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
        return logits, state

    def inference(self, inputs):
        print("????????") # shouldn't reach here
        return None

class Critic(BaseNet):
    def __init__(self,
        preprocess_net,
        output_dim = 1):

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
        print("????????") # shouldn't reach here
        return {}

    def network_forward(self, inputs):
        logits, state = self.preprocess(inputs['obs'])
        logits = self.last(logits)
        return logits

    def inference(self, inputs):
        print("????????") # shouldn't reach here
        return None