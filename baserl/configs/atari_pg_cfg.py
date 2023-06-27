#!/usr/bin/python3
# -*- coding:utf-8 -*-

from loguru import logger
import megengine.distributed as dist
import envpool
import megfile
import numpy as np

from baserl.configs.base_cfg import BaseConfig

class AtariPGConfig(BaseConfig):

    def __init__(self, taskname):
        super().__init__()
        self.DATA.TASK_TYPE = "Atari"
        self.DATA.TASK_NAME = taskname + "NoFrameskip-v4"
        self.DATA.TASK_NAME_ENVPOOL = taskname + "-v5"
        self.DATA.TRAINING_ENV_ARGS = {
            "stack_num": 4 ,
            "episodic_life": True,
            "reward_clip": True,
        }
        self.DATA.TEST_ENV_ARGS = {
            "stack_num": 4 ,
            "episodic_life": False,
            "reward_clip": False,
        }
        self.DATA.BUFFER_NAME = "VectorReplayBuffer"
        self.DATA.BUFFER_SIZE = 100000
        self.DATA.TEST_ENV_NUM = 10
        self.DATA.TRAINING_ENV_NUM = 64
        
        self.MODEL.BACKBONE.TYPE = "CNN"
        self.MODEL.BACKBONE.NAME = "PGNet"
        env = envpool.make_gymnasium(self.DATA.TASK_NAME_ENVPOOL,) 
        state_shape = env.observation_space.shape or env.observation_space.n
        self.MODEL.STATE_SHAPE = np.prod(state_shape)
        action_shape = env.action_space.shape or env.action_space.n
        self.MODEL.ACTION_SHAPE = np.prod(action_shape)
        self.MODEL.BACKBONE.CONFIGURATIONS = {
            "input_channels": self.DATA.TRAINING_ENV_ARGS["stack_num"], # 210x160x3 -> 4x84x84
            "output_dim": self.MODEL.ACTION_SHAPE,
            "hidden_sizes": [32, 64, 64, 3136, 512],
            "kernel_sizes": [8, 4, 3],
            "strides": [4, 2, 1],
            "use_softmax": True,
        }
        self.sample_env = env
        
        self.MODEL.STEP_PER_COLLECT = 2048
        self.MODEL.EPISODE_PER_COLLECT = None # each collector.collect will collect ~ episode
        self.MODEL.REPEAT_PER_COLLECT = 2
        self.MODEL.REWARD_METRIC = None
        self.MODEL.GAMMA = 0.99
        self.MODEL.PG_REWARD_NORM = True
        self.MODEL.POLICY_NAME = "PGPolicy"
        self.MODEL.POLICY_TRAINER_NAME = "OnPolicyTrainer"
        self.MODEL.POLICY_DISTRIBUTION_NAME = "Categorical"
        self.MODEL.TEST_EXPLORATION_NOISE = True
        
        self.TRAINER.BATCH_SIZE = 99999
        self.SOLVER.NUM_ENV_STEP_PER_EPOCH = 100000 # each epoch ends when the total collected steps are more than ~
        self.SOLVER.NUM_ENV_EPISODE_PER_EPOCH = None
        self.SOLVER.BASIC_LR = 1e-4
        self.SOLVER.MAX_EPOCH = 100
        self.SOLVER.OPTIMIZER_NAME = 'Adam'
        self.SOLVER.EXTRA_OPT_ARGS = dict(
        )
        self.SOLVER.WEIGHT_DECAY = 0.
        self.SOLVER.LR_SCHEDULE = "linear"
        self.GLOBAL.LOG_INTERVAL = 3 
        self.GLOBAL.OUTPUT_DIR = f'logs/pg/{self.DATA.TASK_NAME}'
        self.GLOBAL.CKPT_SAVE_DIR = f'/data/Outputs/model_logs/train_pg/{self.DATA.TASK_NAME}'

    def build_model(self):
        import baserl.models as BM
        network_cls = getattr(BM, self.MODEL.BACKBONE.TYPE)
        return network_cls(**self.MODEL.BACKBONE.CONFIGURATIONS)

    def build_dataloader(self, model):
        from baserl.data.provider.collector import Collector
        import baserl.policy as BP
        import baserl.data.provider as BDP
        import baserl.utils as BU
        from baserl.utils import Independent
        policy_cls = getattr(BP, self.MODEL.POLICY_NAME)
        buffer_cls = getattr(BDP, self.DATA.BUFFER_NAME)
        distribution_cls = getattr(BU, self.MODEL.POLICY_DISTRIBUTION_NAME)
        
        def distribution_func(p):
            return distribution_cls(probs=p)
        
        train_envs = envpool.make_gymnasium(
            self.DATA.TASK_NAME_ENVPOOL,
            num_envs=self.DATA.TRAINING_ENV_NUM,
            seed=self.seed,
            **self.DATA.TRAINING_ENV_ARGS,
        )

        policy = policy_cls(
            model,
            dist_fn=distribution_func,
            discount_factor=self.MODEL.GAMMA, 
            reward_normalization=self.MODEL.PG_REWARD_NORM, 
            action_space=self.sample_env.action_space,
        )

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
