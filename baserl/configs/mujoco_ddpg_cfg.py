#!/usr/bin/python3
# -*- coding:utf-8 -*-

from loguru import logger
import megengine.distributed as dist
import envpool
import megfile
import numpy as np

from baserl.models.net_ddpg import ActorDDPG, CriticDDPG
from baserl.models.net_continous import ActorCritic
from baserl.models.mlp import MLP

from baserl.exploration import GaussianNoise
from baserl.configs.base_cfg import BaseConfig

class MujocoDDPGConfig(BaseConfig):

    def __init__(self, taskname):
        super().__init__()
        
        self.GLOBAL.LOG_INTERVAL = 100
        self.SOLVER.NUM_ENV_STEP_PER_EPOCH = 10000
        if "InvertedDoublePendulum" in taskname:
            self.SOLVER.MAX_EPOCH = 20 
        else:
            self.SOLVER.MAX_EPOCH = 100 
        
        self.DATA.TASK_TYPE = "Mujoco"
        self.DATA.TASK_NAME = taskname
        self.DATA.TASK_NAME_ENVPOOL = taskname
        env = envpool.make_gymnasium(self.DATA.TASK_NAME_ENVPOOL,)
        state_shape = env.observation_space.shape or env.observation_space.n
        self.MODEL.STATE_SHAPE = np.prod(state_shape)
        action_shape = env.action_space.shape or env.action_space.n
        self.MODEL.ACTION_SHAPE = np.prod(action_shape)
        self.MODEL.MAX_ACTION = env.action_space.high[0]
        self.sample_env = env
        
        self.MODEL.BACKBONE.TYPE = "MLP"
        self.MODEL.BACKBONE.CONFIGURATIONS_1 = {
            "input_dim": self.MODEL.STATE_SHAPE,
            "hidden_sizes": [256, 256],
        }
        self.MODEL.BACKBONE.CONFIGURATIONS_2 = {
            "input_dim": self.MODEL.STATE_SHAPE+self.MODEL.ACTION_SHAPE,
            "hidden_sizes": [256, 256],
        }
        self.MODEL.BACKBONE.NAME_A = "ActorDDPG"
        self.MODEL.BACKBONE.NAME_C = "CriticDDPG"
        self.MODEL.BACKBONE.CONFIGURATIONS_A = [self.MODEL.ACTION_SHAPE, self.MODEL.MAX_ACTION]
        self.MODEL.BACKBONE.NAME_FULL = "ActorCriticDDPG"

        # network params
        if "Ant" in taskname:
            self.SOLVER.ACTOR_LR = 1e-4
            self.SOLVER.CRITIC_LR = 1e-4
        else:
            self.SOLVER.ACTOR_LR = 1e-3
            self.SOLVER.CRITIC_LR = 1e-3

        # policy params
        self.MODEL.STEP_PER_COLLECT = 10
        # self.step_per_epoch = 20000
        self.MODEL.UPDATE_PER_STEP = 0.1
        # self.episode_per_collect = None
        # self.repeat_per_collect = None
        self.MODEL.REWARD_METRIC = None
        self.TRAINER.BATCH_SIZE = 256
        self.DATA.TRAINING_ENV_NUM = 10
        self.DATA.BUFFER_SIZE = 100000
        self.MODEL.EPS_SCHEDULE = "fixed"
        self.MODEL.EPS_END = 0.1
        self.MODEL.COLLECT_BEFORE_TRAIN = 25000
        self.MODEL.COLLECT_BEFORE_TRAIN_RANDOM = True

        # policy args
        self.MODEL.TAU = 0.005
        self.MODEL.GAMMA = 0.99
        self.MODEL.DDPG_GAUSSIAN_NOISE_STD = 0.1
        self.MODEL.DDPG_REWARD_NORM = False
        self.MODEL.N_STEP = 1
        
        self.GLOBAL.CKPT_SAVE_DIR = f'/data/Outputs/model_logs/train_ddpg/{self.DATA.TASK_NAME}'
        self.GLOBAL.OUTPUT_DIR = f'logs/ddpg/{self.DATA.TASK_NAME}'

    def build_model(self):
        import baserl.models as BM
        bb_cls = getattr(BM, self.MODEL.BACKBONE.TYPE)
        a_cls = getattr(BM, self.MODEL.BACKBONE.NAME_A)
        c_cls = getattr(BM, self.MODEL.BACKBONE.NAME_C)
        full_cls = getattr(BM, self.MODEL.BACKBONE.NAME_FULL)
        net1 = bb_cls(**self.MODEL.BACKBONE.CONFIGURATIONS_1)
        actor = a_cls(net1, *self.MODEL.BACKBONE.CONFIGURATIONS_A)
        net2 = bb_cls(**self.MODEL.BACKBONE.CONFIGURATIONS_2)
        critic = c_cls(net2)

        print(actor)
        print(critic)
        model = full_cls(actor, critic)

        return model

    def build_dataloader(self, model):
        from baserl.policy import DDPGPolicy
        from baserl.data.provider.collector import Collector 
        from baserl.data.provider import VectorReplayBuffer

        train_envs = envpool.make_gymnasium(
            self.DATA.TASK_NAME_ENVPOOL,
            num_envs=self.DATA.TRAINING_ENV_NUM,
            seed=self.seed,
            **self.DATA.TRAINING_ENV_ARGS,
        )

        policy = DDPGPolicy(
            model,
            tau=self.MODEL.TAU,
            gamma=self.MODEL.GAMMA,
            exploration_noise=GaussianNoise(sigma=self.MODEL.DDPG_GAUSSIAN_NOISE_STD),
            reward_normalization=self.MODEL.DDPG_REWARD_NORM,
            estimation_step=self.MODEL.N_STEP,
            action_space=self.sample_env.action_space
        )

        buf = VectorReplayBuffer(self.DATA.BUFFER_SIZE, buffer_num=len(train_envs))

        collector = Collector(policy, train_envs, buf, exploration_noise=True)
        # collector.reset_stat() 
        
        return collector, policy

    def build_solver(self, model):
        from baserl.solver.default_solver import DefaultSolver
        if model is None:
            model = self.build_model()
        return DefaultSolver.build(self, model)


    def build_trainer(self):
        from baserl.engine import OffPolicyDDPGTrainer
        model = self.build_model()

        weights = self.MODEL.WEIGHTS
        if not weights:
            logger.warning("Train model from scratch...")
        else:
            logger.warning("Loading model weights from {}".format(weights))
            with megfile.smart_open(weights, "rb") as f:
                model.load_weights(f)

        # sync parameters
        if dist.get_world_size() > 1:
            dist.bcast_list_(model.parameters(), dist.WORLD)
            dist.bcast_list_(model.buffers(), dist.WORLD)

        dataloader, policy = self.build_dataloader(model)

        solver = {}
        self.SOLVER.BASIC_LR = self.SOLVER.ACTOR_LR
        solver['solver_actor'] = self.build_solver(model.actor)
        self.SOLVER.BASIC_LR = self.SOLVER.CRITIC_LR
        solver['solver_critic'] = self.build_solver(model.critic)

        hookslist = self.build_hooks()

        trainer = OffPolicyDDPGTrainer(self, model, dataloader, solver, hooks=hookslist)

        trainer.policy = policy

        return trainer

    def build_hooks(self):
        from baserl.engine import build_rl_hooks
        return build_rl_hooks(self)
