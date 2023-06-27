# from matplotlib import widgets
from baserl.data.provider import batch
import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np

from baserl import layers
from .base_net import BaseNet
from .net_continous import ActorProb as ActorProbContPG
from .mlp import MLP
from .cnn import CNN

from baserl.layers import safelog

SIGMA_MIN = -20
SIGMA_MAX = 2
