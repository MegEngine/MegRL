#!/usr/bin/python3
# -*- coding:utf-8 -*-
from typing import List
import megfile

from basecore.utils import is_rank0_process, str_timestamp

from .hooks import (
    BaseHook,
    CheckpointHook,
    EvalHook,
    LoggerHook,
    LRSchedulerHook,
    ResumeHook,
    TensorboardHook,
    GenerateVideoHook,
    PlotHook,
)

__all__ = [
    "build_rl_hooks",
]

def build_rl_hooks(cfg) -> List[BaseHook]:
    ckpt_dir = megfile.smart_path_join(cfg.GLOBAL.CKPT_SAVE_DIR, "ckpt")

    hook_list = [LRSchedulerHook(cfg.SOLVER.WARM_ITERS)]
    if cfg.TRAINER.RESUME:
        hook_list.append(ResumeHook(ckpt_dir, cfg.TRAINER.RESUME))

    if is_rank0_process():
        if cfg.GLOBAL.TENSORBOARD.ENABLE:
            # Since LoggerHook will reset value, tb hook should be added before LoggerHook
            tb_dir_with_time = megfile.smart_path_join(
                cfg.GLOBAL.OUTPUT_DIR, "tensorboard", str_timestamp()
            )
            hook_list.append(TensorboardHook(tb_dir_with_time))

        hook_list.append(LoggerHook(cfg.GLOBAL.LOG_INTERVAL))
        hook_list.append(EvalHook(cfg.TEST.EVAL_EPOCH_INTERVAL, eval_times=cfg.DATA.TEST_ENV_NUM))
        hook_list.append(CheckpointHook(ckpt_dir))
        hook_list.append(GenerateVideoHook(cfg.GLOBAL.OUTPUT_DIR,
                                           ckpt_dir, 
                                           is_atari=(cfg.DATA.TASK_TYPE=="Atari")))
        hook_list.append(PlotHook(cfg.GLOBAL.OUTPUT_DIR))

        
    
    return hook_list
