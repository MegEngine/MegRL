import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np
import math

from baserl import layers
from baserl.layers import safelog, linear_init

from .base_net import BaseNet

class CNN(M.Module):
    def __init__(self, input_channels, hidden_sizes, output_dim, 
                 kernel_sizes, strides, use_softmax=False, scale_obs=False):
        super().__init__()
        cnn = [
            M.Conv2d(
                in_channels=input_channels, 
                out_channels=hidden_sizes[0], 
                kernel_size=kernel_sizes[0],
                stride=strides[0]), 
            M.ReLU()]
        for i in range(len(kernel_sizes) - 1):
            cnn += [
                M.Conv2d(
                    in_channels=hidden_sizes[i], 
                    out_channels=hidden_sizes[i+1], 
                    kernel_size=kernel_sizes[i+1],
                    stride=strides[i+1]),
                M.ReLU()]
        self.cnn = M.Sequential(*cnn)
        # self.flatten_dim = np.prod(self.cnn(F.zeros((1, 3, 210, 160))).shape[1:])
        # print(self.flatten_dim)
        # exit()
        mlp = []
        for i in range(len(kernel_sizes), len(hidden_sizes)-1):
            mlp += [
                M.Linear(hidden_sizes[i], hidden_sizes[i+1]), 
                M.ReLU()]
        if output_dim != 0:
            mlp += [M.Linear(hidden_sizes[-1], output_dim)]
            self.output_dim = output_dim
        else:
            self.output_dim = hidden_sizes[-1]
        
        if use_softmax:
            mlp += [M.Softmax(axis=1)]
            
        self.mlp = M.Sequential(*mlp)
        
        self._init_modules()
        self.scale_obs = scale_obs
        
    def _init_modules(self):
        
        for m in self.modules():
            if isinstance(m, M.Linear):
                linear_init(m)
            if isinstance(m, M.Conv2d):
                M.init.msra_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = M.init.calculate_fan_in_and_fan_out(m.weight)
                    if fan_in != 0:
                        bound = 1 / math.sqrt(fan_in)
                        M.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        if self.scale_obs:
            x /= 255.0
        x = self.cnn(x)
        x = F.flatten(x, start_axis=1, end_axis=-1)
        x = self.mlp(x)
        
        return x, None
