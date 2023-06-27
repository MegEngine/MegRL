#!/usr/bin/python3
# -*- coding:utf-8 -*-

from baserl import exploration
from loguru import logger
import megengine.distributed as dist
from megengine.data import DataLoader
from megengine.module import Module
import envpool
import megfile
import gymnasium as gym
import numpy as np
from baserl.exploration import GaussianNoise

from baserl.configs.base_cfg import BaseConfig
from baserl.data.provider.collector import Collector


class ClassicDiscreteSACConfig(BaseConfig):

    def __init__(self, taskname):
        super().__init__()
        self.DATA.TASK_TYPE = "Classic" 
        self.DATA.TASK_NAME = taskname 
        self.DATA.TASK_NAME_ENVPOOL = taskname 
        self.DATA.BUFFER_NAME = "VectorReplayBuffer"
        self.DATA.BUFFER_SIZE = 20000
        self.DATA.TEST_ENV_NUM = 10
        self.DATA.TRAINING_ENV_NUM = 10
        
        self.MODEL.BACKBONE.TYPE = "MLP"
        env = envpool.make_gymnasium(self.DATA.TASK_NAME_ENVPOOL,)
        state_shape = env.observation_space.shape or env.observation_space.n
        self.MODEL.STATE_SHAPE = np.prod(state_shape)
        action_shape = env.action_space.shape or env.action_space.n
        self.MODEL.ACTION_SHAPE = np.prod(action_shape)
        self.MODEL.BACKBONE.CONFIGURATIONS_MLP = {
            "input_dim": self.MODEL.STATE_SHAPE,
            # "output_dim": self.MODEL.ACTION_SHAPE,
            "hidden_sizes": [64, 64],
            # "use_softmax": False,
        }
        self.sample_env = env
        
        self.MODEL.BACKBONE.NAME_A = "ActorSAC"
        self.MODEL.BACKBONE.NAME_C = "CriticSAC"
        self.MODEL.BACKBONE.NAME_FULL = "ActorCriticSAC"
        self.MODEL.POLICY_NAME = "DiscreteSACPolicy"
        self.MODEL.POLICY_TRAINER_NAME = "OffPolicyDiscreteSACTrainer"
        self.MODEL.TAU = 0.005
        self.MODEL.GAMMA = 0.95
        self.MODEL.ALPHA = 0.05
        self.MODEL.SAC_REWARD_NORM = False
        self.MODEL.N_STEP = 3
        self.MODEL.DETERMINISTIC_EVAL = False
        
        self.SOLVER.BASIC_LR = 3e-4
        self.SOLVER.ACTOR_LR = 5e-3
        self.SOLVER.CRITIC1_LR = 5e-3
        self.SOLVER.CRITIC2_LR = 5e-3

        self.SOLVER.MAX_EPOCH = 30
        self.MODEL.STEP_PER_COLLECT = 10
        self.SOLVER.NUM_ENV_STEP_PER_EPOCH = 10000
        self.SOLVER.NUM_ENV_EPISODE_PER_EPOCH = None
        self.MODEL.UPDATE_PER_STEP = 0.1
        self.MODEL.EPISODE_PER_COLLECT = None

        self.TRAINER.BATCH_SIZE = 64
        self.SOLVER.OPTIMIZER_NAME = 'Adam'
        self.SOLVER.EXTRA_OPT_ARGS = dict(
        )
        
        self.GLOBAL.LOG_INTERVAL = 100
        self.GLOBAL.OUTPUT_DIR = f'logs/sac/{self.DATA.TASK_NAME}'
        self.GLOBAL.CKPT_SAVE_DIR = f'/data/Outputs/model_logs/train_sac/{self.DATA.TASK_NAME}'     


    def build_model(self):
        import baserl.models as BM
        bb_cls = getattr(BM, self.MODEL.BACKBONE.TYPE)
        a_cls = getattr(BM, self.MODEL.BACKBONE.NAME_A)
        c_cls = getattr(BM, self.MODEL.BACKBONE.NAME_C)
        full_cls = getattr(BM, self.MODEL.BACKBONE.NAME_FULL)
        
        net1 = bb_cls(**self.MODEL.BACKBONE.CONFIGURATIONS_MLP)
        actor = a_cls(net1, self.MODEL.ACTION_SHAPE, softmax_output=False)
        actor._alpha = self.MODEL.ALPHA
        net_c1 = bb_cls(**self.MODEL.BACKBONE.CONFIGURATIONS_MLP)
        critic1 = c_cls(net_c1, output_dim=self.MODEL.ACTION_SHAPE)
        net_c2 = bb_cls(**self.MODEL.BACKBONE.CONFIGURATIONS_MLP)
        critic2 = c_cls(net_c2, output_dim=self.MODEL.ACTION_SHAPE)

        model = full_cls(actor, critic1, critic2)
        print(model)

        return model

    def build_dataloader(self, model):
        import baserl.policy as BP
        import baserl.data.provider as BDP
        buffer_cls = getattr(BDP, self.DATA.BUFFER_NAME)
        policy_cls = getattr(BP, self.MODEL.POLICY_NAME)

        train_envs = envpool.make_gymnasium(
            self.DATA.TASK_NAME_ENVPOOL,
            num_envs=self.DATA.TRAINING_ENV_NUM,
            seed=self.seed, 
            **self.DATA.TRAINING_ENV_ARGS,
        )

        policy = policy_cls(
            model,
            tau=self.MODEL.TAU,
            gamma=self.MODEL.GAMMA,
            alpha=self.MODEL.ALPHA,
            reward_normalization=self.MODEL.SAC_REWARD_NORM,
            estimation_step=self.MODEL.N_STEP,
            deterministic_eval=self.MODEL.DETERMINISTIC_EVAL,
        )

        buf = buffer_cls(self.DATA.BUFFER_SIZE, buffer_num=len(train_envs))

        collector = Collector(policy, train_envs, buf)
        # collector.reset_stat()

        return collector, policy

    def build_solver(self, model):
        from baserl.solver.default_solver import DefaultSolver
        if model is None:
            model = self.build_model()
        return DefaultSolver.build(self, model)


    def build_trainer(self):
        import baserl.engine as BE
        policy_trainer_cls = getattr(BE, self.MODEL.POLICY_TRAINER_NAME)
        
        logger.info("Using model named {}".format(self.MODEL.BACKBONE.NAME_FULL))
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

        logger.info("Using {} training env".format("envpool"))
        logger.info("Using policy named {}".format(self.MODEL.POLICY_NAME))
        logger.info("Using buffer named {}".format(self.DATA.BUFFER_NAME))
        dataloader, policy = self.build_dataloader(model)

        logger.info("Using solver named {}".format(self.SOLVER.OPTIMIZER_NAME))
        solver = {}
        self.SOLVER.BASIC_LR = self.SOLVER.ACTOR_LR
        solver['solver_actor'] = self.build_solver(model.actor)
        self.SOLVER.BASIC_LR = self.SOLVER.CRITIC1_LR
        solver['solver_critic1'] = self.build_solver(model.critic1)
        self.SOLVER.BASIC_LR = self.SOLVER.CRITIC2_LR
        solver['solver_critic2'] = self.build_solver(model.critic2)

        hookslist = self.build_hooks()

        logger.info("Using policy-trainer named {}".format(self.MODEL.POLICY_TRAINER_NAME))
        trainer = policy_trainer_cls(self, model, dataloader, solver, hooks=hookslist)

        trainer.policy = policy

        return trainer

    def build_hooks(self):
        from baserl.engine import build_rl_hooks
        return build_rl_hooks(self)
