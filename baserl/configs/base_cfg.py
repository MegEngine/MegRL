#!/usr/bin/python3
# -*- coding:utf-8 -*-

import getpass
import os
import megfile
from loguru import logger

import megengine.distributed as dist
from megengine.data import DataLoader
from megengine.module import Module

from basecore.config import ConfigDict
from basecore.utils import ensure_dir

from baserl.configs.extra_cfg import (
    DataConfig,
    GlobalConfig,
    ModelConfig,
    SolverConfig,
    TestConfig,
    TrainerConfig,
)


class BaseConfig(ConfigDict):

    user = getpass.getuser()

    def __init__(self, cfg=None, **kwargs):
        """
        params in kwargs is the latest value
        """
        super().__init__(cfg, **kwargs)
        self.MODEL: ModelConfig = ModelConfig()
        self.DATA: DataConfig = DataConfig()
        self.SOLVER: SolverConfig = SolverConfig()

        # training
        self.TRAINER: TrainerConfig = TrainerConfig()
        # testing
        self.TEST: TestConfig = TestConfig()
        self.AUG = dict()
        self.GLOBAL: GlobalConfig = GlobalConfig()

    def link_log_dir(self, link_name="log"):
        """
        create soft link to output dir.

        Args:
            link_name (str): name of soft link.
        """
        output_dir = self.get("output_dir", None)
        if not output_dir:
            output_dir = self.GLOBAL.OUTPUT_DIR

        if not output_dir:
            raise ValueError("output dir is not specified")
        ensure_dir(output_dir)

        if os.path.islink(link_name) and os.readlink(link_name) != output_dir:
            os.system("rm " + link_name)
        if not os.path.exists(link_name):
            cmd = "ln -s {} {}".format(output_dir, link_name)
            os.system(cmd)

    def build_model(self) -> Module:
        raise NotImplementedError

    def build_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def build_solver(self, model):
        raise NotImplementedError

    def build_trainer(self):
        from baserl.engine import Trainer
        logger.info("Using model named {}".format(self.MODEL.NAME))
        model = self.build_model()

        weights = self.MODEL.WEIGHTS
        if not weights:
            logger.warning("Train model from scrach...")
        else:
            logger.info("Loading model weights from {}".format(weights))
            with megfile.smart_open(weights, "rb") as f:
                model.load_weights(f)

        # sync parameters
        if dist.get_world_size() > 1:
            dist.bcast_list_(model.parameters(), dist.WORLD)
            dist.bcast_list_(model.buffers(), dist.WORLD)

        logger.info("Using dataloader named {}".format(self.DATA.BUILDER_NAME))
        dataloader = self.build_dataloader()

        solver_builder_name = self.SOLVER.BUILDER_NAME
        logger.info("Using solver named {}".format(solver_builder_name))
        solver = self.build_solver(model)

        hooks_builder_name = self.HOOKS.BUILDER_NAME
        logger.info("Using hook list named {}".format(hooks_builder_name))
        hookslist = self.build_hooks()

        trainer_name = self.TRAINER.NAME
        logger.info("Using trainer named {}".format(trainer_name))
        trainer = Trainer(self, model, dataloader, solver, hooks=hookslist)
        return trainer

    def build_evaluator(self):
        raise NotImplementedError

    def build_hooks(self):
        from baserl.engine import build_simple_hooks
        return build_simple_hooks(self)
