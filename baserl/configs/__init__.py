#!/usr/bin/python3
# -*- coding:utf-8 -*-

from basecore.config import ConfigDict

from .base_cfg import BaseConfig
from .classic_dqn_cfg import ClassicDQNConfig
from .classic_ddpg_cfg import ClassicDDPGConfig
from .classic_pg_cfg import ClassicPGConfig, ClassicContPGConfig
from .classic_a2c_cfg import ClassicA2CConfig, ClassicContA2CConfig
from .classic_ppo_cfg import ClassicPPOConfig, ClassicContPPOConfig
from .classic_discrete_sac_cfg import ClassicDiscreteSACConfig
from .classic_continuous_sac_cfg import ClassicContinuousSACConfig

from .atari_dqn_cfg import AtariDQNConfig
from .atari_pg_cfg import AtariPGConfig
from .atari_a2c_cfg import AtariA2CConfig
from .atari_ppo_cfg import AtariPPOConfig
from .atari_discrete_sac_cfg import AtariDiscreteSACConfig

from .mujoco_ddpg_cfg import MujocoDDPGConfig
from .mujoco_pg_cfg import MujocoPGConfig
from .mujoco_a2c_cfg import MujocoA2CConfig
from .mujoco_ppo_cfg import MujocoPPOConfig
from .mujoco_continuous_sac_cfg import MujocoContinuousSACConfig

# from .extra_cfg import (
#     DataConfig,
#     GlobalConfig,
#     ModelConfig,
#     SolverConfig,
#     TestConfig,
#     TrainerConfig,
# )

__all__ = [k for k in globals().keys() if not k.startswith("_")]
