#!/usr/bin/python3
# -*- coding:utf-8 -*-
from loguru import logger
import megengine.distributed as dist
import envpool
import megfile
import megengine.module as M
import gymnasium as gym
import numpy as np

from baserl.configs.base_cfg import BaseConfig

class MujocoPPOConfig(BaseConfig):

    def __init__(self, taskname):
        super().__init__()
        self.DATA.TASK_TYPE = "Mujoco"
        self.DATA.TASK_NAME = taskname
        self.DATA.TASK_NAME_ENVPOOL = taskname
        self.DATA.BUFFER_NAME = "VectorReplayBuffer"
        self.DATA.BUFFER_SIZE = 4096
        self.DATA.TEST_ENV_NUM = 10
        self.DATA.TRAINING_ENV_NUM = 64
        
        self.MODEL.BACKBONE.TYPE = "MLP"
        self.MODEL.BACKBONE.NAME_A = "ActorProbContPPO"
        self.MODEL.BACKBONE.NAME_C = "CriticContPPO"
        self.MODEL.BACKBONE.NAME_FULL = "ActorCriticPPOCont"
        env = envpool.make_gymnasium(self.DATA.TASK_NAME_ENVPOOL,)
        state_shape = env.observation_space.shape or env.observation_space.n
        self.MODEL.STATE_SHAPE = np.prod(state_shape)
        action_shape = env.action_space.shape or env.action_space.n
        self.MODEL.ACTION_SHAPE = np.prod(action_shape)
        self.MODEL.BACKBONE.CONFIGURATIONS_MLP = {
            "input_dim": self.MODEL.STATE_SHAPE,
            # "output_dim": self.MODEL.ACTION_SHAPE,
            "hidden_sizes": [256, 256],
            # "use_softmax": False,
            "activation": "tanh",
        }
        self.sample_env = env
        
        self.MODEL.POLICY_NAME = "PPOPolicy"
        self.MODEL.POLICY_TRAINER_NAME = "OnPolicyTrainer"
        self.MODEL.POLICY_DISTRIBUTION_NAME = "Normal"
        self.MODEL.GAMMA = 0.99
        self.MODEL.PPO_REWARD_NORM = 1
        self.MODEL.SIGMA_INIT = -0.5
        # ppo special
        self.MODEL.PPO_VALUE_FUNC_COEFF = 0.25
        self.MODEL.PPO_ENTROPY_COEFF = 0.0
        self.MODEL.PPO_EPS_CLIP = 0.2
        self.MODEL.PPO_GAE_LAMBDA = 0.95
        self.MODEL.PPO_NORM_ADV = 0
        self.MODEL.PPO_RECOMPUTE_ADV = 1
        self.MODEL.PPO_DETERMINISTIC_EVAL = False
        self.MODEL.PPO_DUAL_CLIP = None
        self.MODEL.PPO_VALUE_CLIP = 0
        self.MODEL.ACTION_BOUND_METHOD = "clip"
        
        
        self.SOLVER.BASIC_LR = 5e-4
        self.SOLVER.LR_SCHEDULE = "linear"
        self.TRAINER.GRAD_CLIP = dict(
            ENABLE=True,
            # supported type: ("value", "norm")
            TYPE="norm",
            # ARGS=dict(lower=-1, upper=1)
            ARGS=dict(max_norm=0.5, ord=2)
        )
        self.SOLVER.MAX_EPOCH = 150
        self.MODEL.STEP_PER_COLLECT = 2048
        self.MODEL.EPISODE_PER_COLLECT = None
        self.SOLVER.NUM_ENV_STEP_PER_EPOCH = 30000
        self.SOLVER.NUM_ENV_EPISODE_PER_EPOCH = None
        self.MODEL.REPEAT_PER_COLLECT = 10
        self.TRAINER.BATCH_SIZE = 64
        self.SOLVER.OPTIMIZER_NAME = 'Adam'
        self.SOLVER.EXTRA_OPT_ARGS = dict(
        )
        self.GLOBAL.LOG_INTERVAL = 1

        self.GLOBAL.OUTPUT_DIR = f'logs/ppo/{self.DATA.TASK_NAME}'
        self.GLOBAL.CKPT_SAVE_DIR = f'/data/Outputs/model_logs/train_ppo/{self.DATA.TASK_NAME}'

    def build_model(self):
        import baserl.models as BM
        bb_cls = getattr(BM, self.MODEL.BACKBONE.TYPE)
        a_cls = getattr(BM, self.MODEL.BACKBONE.NAME_A)
        c_cls = getattr(BM, self.MODEL.BACKBONE.NAME_C)
        full_cls = getattr(BM, self.MODEL.BACKBONE.NAME_FULL)
        
        preprocess_net_a = bb_cls(**self.MODEL.BACKBONE.CONFIGURATIONS_MLP)
        preprocess_net_c = bb_cls(**self.MODEL.BACKBONE.CONFIGURATIONS_MLP)
        actor = a_cls(
            preprocess_net=preprocess_net_a,
            hidden_sizes=(),
            output_dim=self.MODEL.ACTION_SHAPE,
            unbounded=True,
        )
        critic = c_cls(preprocess_net_c)
        model = full_cls(actor, critic)
        M.init.fill_(actor.sigma_param, self.MODEL.SIGMA_INIT)
        for m in actor.mu.modules():
            if isinstance(m, M.Linear):
                m.bias *= 0.01
                m.weight *= 0.01

        return model

    def build_dataloader(self, model):
        from baserl.data.provider.collector import Collector
        import baserl.policy as BP
        import baserl.data.provider as BDP
        import baserl.utils as BU
        from baserl.utils import Independent
        policy_cls = getattr(BP, self.MODEL.POLICY_NAME)
        buffer_cls = getattr(BDP, self.DATA.BUFFER_NAME)
        distribution_cls = getattr(BU, self.MODEL.POLICY_DISTRIBUTION_NAME)
        
        def distribution_func(*logits):
            return Independent(distribution_cls(*logits), 1)
        
        train_envs = envpool.make_gymnasium(
            self.DATA.TASK_NAME_ENVPOOL,
            num_envs=self.DATA.TRAINING_ENV_NUM,
            seed=self.seed,
            **self.DATA.TRAINING_ENV_ARGS,
        )

        policy = policy_cls(
            model.actor,
            model.critic,
            dist_fn=distribution_func,
            discount_factor=self.MODEL.GAMMA,
            eps_clip=self.MODEL.PPO_EPS_CLIP,
            vf_coef=self.MODEL.PPO_VALUE_FUNC_COEFF,
            ent_coef=self.MODEL.PPO_ENTROPY_COEFF,
            gae_lambda=self.MODEL.PPO_GAE_LAMBDA,
            reward_normalization=self.MODEL.PPO_REWARD_NORM,
            dual_clip=self.MODEL.PPO_DUAL_CLIP,
            value_clip=self.MODEL.PPO_VALUE_CLIP,
            action_space=self.sample_env.action_space,
            deterministic_eval=self.MODEL.PPO_DETERMINISTIC_EVAL,
            advantage_normalization=self.MODEL.PPO_NORM_ADV,
            recompute_advantage=self.MODEL.PPO_RECOMPUTE_ADV,
            
            action_bound_method=self.MODEL.ACTION_BOUND_METHOD,
            action_scaling=True,
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
