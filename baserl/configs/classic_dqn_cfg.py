#!/usr/bin/python3
# -*- coding:utf-8 -*-

from loguru import logger
import megengine.distributed as dist
import megfile
import numpy as np
import envpool
import math

from baserl.configs.base_cfg import BaseConfig


class ClassicDQNConfig(BaseConfig):

    def __init__(self, taskname):
        super().__init__()
        self.DATA.TASK_TYPE = "Classic"
        self.DATA.TASK_NAME = taskname 
        self.DATA.TASK_NAME_ENVPOOL = taskname
        self.DATA.BUFFER_NAME = "VectorReplayBuffer"
        self.DATA.BUFFER_SIZE = 10000
        
        self.MODEL.BACKBONE.TYPE = "MLP"
        self.MODEL.BACKBONE.NAME = "DQNNet"
        env = envpool.make_gymnasium(self.DATA.TASK_NAME_ENVPOOL,)
        state_shape = env.observation_space.shape or env.observation_space.n
        self.MODEL.STATE_SHAPE = np.prod(state_shape)
        action_shape = env.action_space.shape or env.action_space.n
        self.MODEL.ACTION_SHAPE = np.prod(action_shape)
        self.MODEL.BACKBONE.CONFIGURATIONS = {
            "input_dim": self.MODEL.STATE_SHAPE,
            "output_dim": self.MODEL.ACTION_SHAPE,
            "hidden_sizes": [128, 128],
        }
        
        self.MODEL.POLICY_TRAINER_NAME = "OffPolicyTrainer"
        self.MODEL.POLICY_NAME = "DQNPolicy"
        self.MODEL.N_STEP = 3
        self.MODEL.GAMMA = 0.99
        self.MODEL.TARGET_FREQ = 500
        self.MODEL.TEST_EPS = 0.
        self.MODEL.EPS_BEGIN = 1.0
        self.MODEL.EPS_END = 0.1
        self.MODEL.EPS_SCHEDULE = "fixed"
        self.MODEL.COLLECT_BEFORE_TRAIN = 0
        
        if "CartPole" in taskname:
            self.SOLVER.BASIC_LR = 1e-4
        elif "Acrobot" in taskname:
            self.SOLVER.BASIC_LR = 2e-4
        else:
            self.SOLVER.BASIC_LR = 1e-4
        self.SOLVER.MAX_EPOCH = 50
        self.SOLVER.NUM_ENV_STEP_PER_EPOCH = 10000 # 1 epoch = ~ iterations; calc reward every epoch
        self.GLOBAL.LOG_INTERVAL = 100 # calc loss every ~ iterations
        self.MODEL.STEP_PER_COLLECT = 10
        self.MODEL.UPDATE_PER_STEP = 0.1
        # (UPDATE_PER_STEP * STEP_PER_COLLECT) train_step after STEP_PER_COLLECT env_step
        # train_step in each epoch = UPDATE_PER_STEP * NUM_ENV_STEP_PER_EPOCH
        self.MODEL.REWARD_METRIC = None
        self.TRAINER.BATCH_SIZE = 128
        
        self.GLOBAL.CKPT_SAVE_DIR = f'/data/Outputs/model_logs/train_dqn/{self.DATA.TASK_NAME}'
        self.GLOBAL.OUTPUT_DIR = f'logs/dqn/{self.DATA.TASK_NAME}'

    def build_model(self):
        import baserl.models as BM
        network_cls = getattr(BM, self.MODEL.BACKBONE.NAME)
        return network_cls(
            model_type=self.MODEL.BACKBONE.TYPE,
            model_config=self.MODEL.BACKBONE.CONFIGURATIONS,
        )

    def build_dataloader(self, model):
        from baserl.data.provider.collector import Collector
        import baserl.policy as BP
        import baserl.data.provider as BDP
        policy_cls = getattr(BP, self.MODEL.POLICY_NAME)
        buffer_cls = getattr(BDP, self.DATA.BUFFER_NAME)
        
        train_envs = envpool.make_gymnasium(
            self.DATA.TASK_NAME_ENVPOOL,
            num_envs=self.DATA.TRAINING_ENV_NUM,
            seed=self.seed,
            **self.DATA.TRAINING_ENV_ARGS,
        )
        policy = policy_cls(model, self.MODEL.GAMMA, self.MODEL.N_STEP, 
                            target_update_freq=self.MODEL.TARGET_FREQ)
        buf = buffer_cls(self.DATA.BUFFER_SIZE, buffer_num=len(train_envs))

        collector = Collector(policy, train_envs, buf, exploration_noise=True)
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
        
        logger.info("Using model named {}".format(self.MODEL.BACKBONE.NAME))
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
        solver = self.build_solver(model)

        hookslist = self.build_hooks()

        logger.info("Using policy-trainer named {}".format(self.MODEL.POLICY_TRAINER_NAME))
        trainer = policy_trainer_cls(self, model, dataloader, solver, hooks=hookslist)

        trainer.policy = policy

        return trainer

    def build_hooks(self):
        from baserl.engine import build_rl_hooks
        return build_rl_hooks(self)
