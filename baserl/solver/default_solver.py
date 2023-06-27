#!/usr/bin/python3
# -*- coding:utf-8 -*-
import math
import megengine.distributed as dist
import megengine.module as M
import megengine.optimizer as optim
from megengine.autodiff import GradManager

from basecore.engine import Solver

__all__ = ["DefaultSolver"]


class DefaultSolver:

    @classmethod
    def build(cls, cfg, model: M.Module) -> Solver:
        """build default solver by provided model and configuration."""
        solver_cfg = cfg.SOLVER
        cls.reduce_mode = solver_cfg.get("REDUCE_MODE", "MEAN")
        assert cls.reduce_mode in ["MEAN", "SUM"]

        optimizer = cls.build_optimizer(cfg, model)
        gm = cls.build_grad_manager(cfg, model)
        scaler = cls.build_grad_scaler(cfg)
        return Solver(optimizer=optimizer, grad_manager=gm, grad_scaler=scaler)

    @classmethod
    def build_optimizer(cls, cfg, model: M.Module):
        """build optimizer to optimize model parameters"""
        lr = cfg.SOLVER.BASIC_LR * cfg.MODEL.BATCHSIZE_PER_DEVICE
        wd = cfg.SOLVER.WEIGHT_DECAY
        world_size = dist.get_world_size()
        if cls.reduce_mode == "MEAN":
            # lr = lr * math.log2(world_size) # FIXME
            pass
        else:
            lr = lr / world_size # FIXME
            wd = wd * world_size # FIXME
        optimizer_name = cfg.SOLVER.get("OPTIMIZER_NAME", "Adam")
        extra_args = cfg.SOLVER.get("EXTRA_OPT_ARGS", dict())
        optimizer = getattr(optim, optimizer_name)(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            **extra_args,
        )
        return optimizer

    @classmethod
    def build_grad_manager(cls, cfg, model: M.Module) -> GradManager:
        gm = GradManager()
        world_size = dist.get_world_size()
        callbacks = [dist.make_allreduce_cb(cls.reduce_mode, dist.WORLD)] if world_size > 1 else None  # noqa
        gm.attach(model.parameters(), callbacks=callbacks)
        return gm

    @classmethod
    def build_grad_scaler(cls, cfg):
        amp_cfg = cfg.TRAINER.AMP
        if amp_cfg.ENABLE:
            from megengine.amp import GradScaler
            scaler = (
                GradScaler(init_scale=65536.0, growth_interval=2000)
                if amp_cfg.DYNAMIC_SCALE
                else GradScaler(init_scale=128.0, growth_interval=0)
            )
            return scaler
        else:
            return None
