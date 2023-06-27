# from matplotlib import widgets
import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np

from baserl import layers
from baserl.layers import safelog, linear_init

from .base_net import BaseNet


class TanH(M.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return F.tanh(x)

class MLP(M.Module):
    def __init__(self, input_dim, 
            hidden_sizes = [], 
            output_dim = 0, 
            use_softmax = False,
            activation="relu"):
        super().__init__()
        model = []
        if len(hidden_sizes) > 0:
            if activation == "relu":
                model += [M.Linear(input_dim, hidden_sizes[0]),
                    M.ReLU()]
            elif activation == "tanh":
                model += [M.Linear(input_dim, hidden_sizes[0]),
                    TanH()]
            else:
                raise NotImplementedError
                
            for i in range(len(hidden_sizes)-1):
                if activation == "relu":
                    model += [M.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                        M.ReLU()]
                elif activation == "tanh":
                    model += [M.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                        TanH()]

        if output_dim == 0:
            self.output_dim = hidden_sizes[-1]
        else:
            if len(hidden_sizes) > 0:
                model += [M.Linear(hidden_sizes[-1], output_dim)]
            else:
                model += [M.Linear(input_dim, output_dim)]
            self.output_dim = output_dim

        if use_softmax:
            model += [M.Softmax(axis=1)]
            
        self.model = M.Sequential(*model)
        self._init_modules()
        self.a=1

    def _init_modules(self):
        for m in self.modules():
            if isinstance(m, M.Linear):
                linear_init(m)

    def forward(self, x):
        x = self.model(x)
        return x, None
