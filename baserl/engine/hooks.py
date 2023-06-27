#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import bisect
import datetime
import math
import os
import time
import matplotlib.pyplot as plt

from numpy import isin
import numpy as np
from loguru import logger

from tensorboardX import SummaryWriter

from basecore.engine import BaseHook

from baserl.utils import (
    Checkpoint,
    cached_property,
    MeterBuffer,
    ensure_dir,
    get_env_info_table,
    get_last_call_deltatime,
    save_video,
)


import envpool
import megengine.distributed as dist
from baserl.data.env.wrapper import wrap_deepmind

__all__ = [
    "BaseHook",
    "LoggerHook",
    "LRSchedulerHook",
    "CheckpointHook",
    "ResumeHook",
    "TensorboardHook",
    "GenerateVideoHook",
]


class LoggerHook(BaseHook):
    """
    Hook to log information with logger.

    NOTE: LoggerHook will clear all values in meters, so be careful about the usage.
    """
    def __init__(self, log_interval=20):
        """
        Args:
            log_interval (int): iteration interval between two logs.
        """
        self.log_interval = log_interval
        self.meter = MeterBuffer(self.log_interval)

    def before_train(self):
        logger.info("\nSystem env:\n{}".format(get_env_info_table()))

        # logging model
        logger.info("\nModel structure:\n" + repr(self.trainer.model))

        # logging config
        cfg = self.trainer.cfg
        logger.info("\nTraining full config:\n" + repr(cfg))

        logger.info(
            "Starting training from epoch {}, iteration {}".format(
                self.trainer.progress.epoch, self.trainer.progress.iter
            )
        )

        self.start_training_time = time.time()

    def after_train(self):
        total_training_time = time.time() - self.start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f} s / iter)".format(
                total_time_str, self.meter["iters_time"].global_avg
            )
        )

    def before_iter(self):
        self.iter_start_time = time.time()

    def after_iter(self):
        single_iter_time = time.time() - self.iter_start_time

        delta_time = get_last_call_deltatime()
        if delta_time is None:
            delta_time = single_iter_time

        prev_iter_len = self.trainer.__dict__.get("prev_iter_len", 1)
        self.meter.update({
            "iters_time": {"value": single_iter_time, "cnt": prev_iter_len}, # to get global average iter time
            "eta_iter_time": {"value": delta_time, "cnt": prev_iter_len},  # to get ETA time
            "extra_time": {"value": delta_time - single_iter_time, "cnt": prev_iter_len},  # to get extra time
        })

        trainer = self.trainer
        epoch_id, iter_id = trainer.progress.epoch, trainer.progress.iter
        max_epoch, max_iter = trainer.progress.max_epoch, trainer.progress.max_iter

        if iter_id % self.log_interval == 0 or (iter_id == 1 and epoch_id == 1):
            log_str_list = []
            # step info string
            log_str_list.append(str(trainer.progress))

            # loss string
            log_str_list.append(self.get_loss_str(trainer.meter))

            # extra logging meter in model
            extra_str = self.get_extra_str()
            if extra_str:
                log_str_list.append(extra_str)

            # other training info like learning rate.
            log_str_list.append(self.get_train_info_str())

            # memory useage.
            # TODO refine next 3lins logic after mge works
            mem_str = self.get_memory_str(trainer.meter)
            if mem_str:
                log_str_list.append(mem_str)

            # time string
            left_iters = max_iter - iter_id + (max_epoch - epoch_id) * max_iter
            time_str = self.get_time_str(left_iters)
            log_str_list.append(time_str)

            log_str = ", ".join(log_str_list)
            logger.info(log_str)
            
            if hasattr(self.trainer.policy, "eps"):
                logger.info(f"Current policy eps: {self.trainer.policy.eps}")

            # reset meters in trainer & model every #log_interval iters
            trainer.meter.reset()
            if hasattr(trainer.model, "extra_meter"):
                trainer.model.extra_meter.reset()

    def get_loss_str(self, meter):
        """Get loss information during trainging process"""
        loss_dict = meter.get_filtered_meter(filter_key="loss")
        loss_str = ", ".join([
            "{}:{:.3f}({:.3f})".format(name, value.latest, value.avg)
            for name, value in loss_dict.items()
        ])
        return loss_str

    def get_memory_str(self, meter):
        """Get memory information during trainging process"""

        def mem_in_Mb(mem_value):
            return int(mem_value / 1024 / 1024)
        mem_dict = meter.get_filtered_meter(filter_key="memory")
        mem_str = ", ".join([
            "{}:{}({})Mb".format(name, mem_in_Mb(value.latest), mem_in_Mb(value.avg))
            for name, value in mem_dict.items()
        ])
        return mem_str

    def get_train_info_str(self):
        """Get training process related information such as learning rate."""
        # extra info to display, such as learning rate
        trainer = self.trainer
        if isinstance(trainer.solver, dict):
            lr_str = ""
            for k, sol in trainer.solver.items():
                lr = sol.optimizer.param_groups[0]["lr"]
                lr_str += "{}_lr:{:.3e},".format(k, lr)
        else:
            lr = trainer.solver.optimizer.param_groups[0]["lr"]
            lr_str = "lr:{:.3e}".format(lr)
        return lr_str

    def get_time_str(self, left_iters):
        """Get time related information sucn as data_time, train_time, ETA and so on."""
        trainer = self.trainer
        time_dict = trainer.meter.get_filtered_meter(filter_key="time")
        train_time_str = ", ".join([
            "{}:{:.3f}s".format(name, value.avg)
            for name, value in time_dict.items()
        ])
        # extra time is stored in loggerHook
        train_time_str += ", extra_time:{:.3f}s, ".format(self.meter["extra_time"].avg)

        eta_seconds = self.meter["eta_iter_time"].global_avg * left_iters
        eta_string = "ETA:{}".format(datetime.timedelta(seconds=int(eta_seconds)))
        time_str = train_time_str + eta_string
        return time_str

    def get_extra_str(self):
        """Get extra information provided by model."""
        # extra_meter is defined in BaseNet
        model = self.trainer.model
        extra_str_list = []
        if hasattr(model, "extra_meter"):
            for key, value in model.extra_meter.items():
                if isinstance(value.latest, str):
                    # non-number types like string
                    formatted_str = "{}:{}".format(key, value.latest)
                elif isinstance(value.latest, int):
                    formatted_str = "{}:{}".format(key, value.latest)
                else:
                    formatted_str = "{}:{:.3f}({:.3f})".format(
                        key, float(value.latest), float(value.avg)
                    )

                extra_str_list.append(formatted_str)

        return ", ".join(extra_str_list)


class LRSchedulerHook(BaseHook):
    """
    Hook to adjust solver learning rate.
    """

    def __init__(self, warm_iters=0):
        """
        Args:
            warm_iters (int): warm up iters for the frist training epoch.
        """
        self.warm_iters = warm_iters

    def before_train(self):
        self.total_lr  # cache learning rate at the begining

    def before_iter(self):
        lr_factor = self.get_lr_factor()
        if isinstance(self.trainer.solver, dict):
            idx = 0
            for k, sol in self.trainer.solver.items():
                pgs = sol.optimizer.param_groups
                for param_group in pgs:
                    param_group["lr"] = lr_factor * self.total_lr[idx]
                idx += 1

        else:
            pgs = self.trainer.solver.optimizer.param_groups
            # adjust lr for optimzer
            for param_group, lr in zip(pgs, self.total_lr):
                param_group["lr"] = lr_factor * lr

    def get_lr_factor(self):
        trainer = self.trainer
        cfg = trainer.cfg
        lr_schedule = cfg.SOLVER.get("LR_SCHEDULE", "step")

        epoch_id, iter_id = trainer.progress.epoch, trainer.progress.iter
        max_epoch, max_iter = trainer.progress.max_epoch, trainer.progress.max_iter

        total_iter = max_epoch * max_iter - self.warm_iters
        cur_iter = (epoch_id - 1) * max_iter + iter_id
        left_iter = max_epoch * max_iter - (cur_iter - 1)

        # Warm up in num_iters at first epoch
        if cur_iter <= self.warm_iters:
            lr_factor = float(cur_iter / self.warm_iters)
        elif lr_schedule == "step":
            decay_rate, decay_stages = cfg.SOLVER.LR_DECAY_RATE, cfg.SOLVER.LR_DECAY_STAGES
            lr_factor = decay_rate ** bisect.bisect_left(decay_stages, epoch_id)
        elif lr_schedule == "linear":
            lr_factor = left_iter / total_iter
        elif lr_schedule == "step_linear":
            if epoch_id <= cfg.SOLVER.LR_DECAY_STAGES:
                lr_factor = 1.0
            else:
                lr_factor = left_iter / (total_iter-cfg.SOLVER.LR_DECAY_STAGES)
        elif lr_schedule == "cosine":
            lr_factor = 0.5 * (1 + math.cos(math.pi * ((total_iter - left_iter) / total_iter)))
        elif lr_schedule is None:
            lr_factor = 1.
        else:
            raise NotImplementedError

        return lr_factor

    @cached_property
    def total_lr(self):
        total_lr_list = []

        if isinstance(self.trainer.solver, dict):
            for k, sol in self.trainer.solver.items():
                param_groups = sol.optimizer.param_groups
                for pg in param_groups:
                    total_lr_list.append(pg["lr"])
        else:
            param_groups = self.trainer.solver.optimizer.param_groups
            for pg in param_groups:
                total_lr_list.append(pg["lr"])
        return total_lr_list


class EvalHook(BaseHook):
    """
    Hook to evalutate model during training process.
    """
    def __init__(self, eval_epoch_interval=None, eval_times=10):
        self.eval_interval = eval_epoch_interval
        self.eval_times = eval_times

    def after_epoch(self):
        trainer = self.trainer
        from baserl.data.provider.collector import Collector
        policy = trainer.policy
        policy.eval()
        if hasattr(policy, "set_eps"):
            policy.set_eps(trainer.cfg.MODEL.TEST_EPS)
        # env = DummyVectorEnv(
        #     [lambda: gym.make(trainer.task_name, render_mode='rgb_array'),], 
        #     seed=self.trainer.cfg.seed)
        env = envpool.make_gymnasium(
            trainer.cfg.DATA.TASK_NAME_ENVPOOL,
            num_envs=self.eval_times,
            seed=trainer.cfg.seed,
            **trainer.cfg.DATA.TEST_ENV_ARGS,
        )
        
        collector = Collector(policy, env, exploration_noise=trainer.cfg.MODEL.get("TEST_EXPLORATION_NOISE", False))
        result = collector.collect(n_episode=self.eval_times, render=0.)
        logger.info(f"Epoch {self.trainer.progress.epoch} Reward: {result['rew']}±{result['rew_std']}")
        trainer.policy.train()
        p = self.trainer.progress
        epoch_test_reward_dict = {
            "epoch_test_reward_mean": result['rew'],
            "epoch_test_reward_max": result['rews'].max(),
            "epoch_test_reward_min": result['rews'].min(),
            "epoch_test_reward_std": result['rew_std'],
            "epoch_test_reward_iter": (p.epoch - 1) * p.max_iter + p.iter - 1
        }
        trainer.reward_meter.update(**epoch_test_reward_dict)

    def after_train(self):
        # self.eval()
        pass
    

class CheckpointHook(BaseHook):

    def __init__(self, save_dir=None, save_period=10):
        ensure_dir(save_dir)
        self.save_dir = save_dir
        self.save_period = save_period
        self.best_epoch_idx = None

    def after_epoch(self):
        trainer = self.trainer
        epoch_id = trainer.progress.epoch

        if isinstance(trainer.solver, dict):
            ckpt_content = {
                "progress": trainer.progress,
            }
            for k, sol in trainer.solver.items():
                ckpt_content["{}_optimizer".format(k)] = sol.optimizer
        else:
            ckpt_content = {
                "optimizer": trainer.solver.optimizer,
                "progress": trainer.progress,
            }
        if trainer.enable_ema:
            ckpt_content["ema"] = trainer.ema

        ckpt = Checkpoint(self.save_dir, trainer.model, **ckpt_content)
        ckpt.save("latest.pkl")
        logger.info("save checkpoint latest.pkl to {}".format(self.save_dir))

        if epoch_id % self.save_period == 0:
            progress_str = trainer.progress.progress_str_list()
            save_name = "_".join(progress_str[:-1]) + ".pkl"
            ckpt.save(save_name)
            logger.info("save checkpoint {} to {}".format(save_name, self.save_dir))
        
        if trainer.reward_meter["epoch_test_reward_mean"]._count == 1 or trainer.reward_meter["epoch_test_reward_mean"].latest >= max(list(trainer.reward_meter["epoch_test_reward_mean"]._deque)):
            ckpt.save("best.pkl")
            logger.info("save checkpoint [BEST] to {}".format(self.save_dir))
            self.best_epoch_idx = trainer.reward_meter["epoch_test_reward_mean"]._count

    def after_train(self):
        logger.info("The best return is {}±{}".format(
            list(self.trainer.reward_meter["epoch_test_reward_mean"]._deque)[self.best_epoch_idx-1], 
            list(self.trainer.reward_meter["epoch_test_reward_std"]._deque)[self.best_epoch_idx-1]))


class ResumeHook(BaseHook):

    def __init__(self, save_dir=None, resume=False):
        ensure_dir(save_dir)
        self.save_dir = save_dir
        self.resume = resume

    def before_train(self):
        trainer = self.trainer
        if self.resume:
            model = trainer.model

            if isinstance(trainer.solver, dict):
                resume_content = {
                    "progress": trainer.progress,
                }
                for k, sol in trainer.solver.items():
                    resume_content["{}_optimizer".format(k)] = sol.optimizer
            else:
                resume_content = {
                    "optimizer": trainer.solver.optimizer,
                    "progress": trainer.progress,
                }
            if trainer.enable_ema:
                resume_content["ema"] = trainer.ema
            ckpt = Checkpoint(self.save_dir, model, **resume_content)
            filename = ckpt.get_checkpoint_file()
            logger.info("load checkpoint from {}".format(filename))
            ckpt.resume()
            # since ckpt is dumped after every epoch,
            # resume training requires epoch + 1 and set iter to 1
            self.trainer.progress.epoch += 1
            self.trainer.progress.iter = 1


class TensorboardHook(BaseHook):

    def __init__(self, log_dir, log_interval=20, scalar_type="latest"):
        """
        Args:
            log_dir (str):
            meter_type (str): support values: "latest", "avg", "global_avg", "median"
        """
        assert scalar_type in ("latest", "avg", "global_avg", "median")
        super().__init__()
        ensure_dir(log_dir)
        self.log_dir = log_dir
        self.type = scalar_type
        self.log_interval = log_interval

    def create_writer(self):
        return SummaryWriter(self.log_dir)

    def before_train(self):
        self.writer = self.create_writer()

    def after_train(self):
        self.writer.close()

    def after_iter(self):
        trainer = self.trainer
        epoch_id, iter_id = trainer.progress.epoch, trainer.progress.iter
        if iter_id % self.log_interval == 0 or (iter_id == 1 and epoch_id == 1):
            self.write(context=trainer)

    def write(self, context):
        cur_iter = self.calcute_iteration(context.progress)
        for key, meter in context.meter.items():
            value = getattr(meter, self.type, None)
            if value is None:
                value = meter.latest
            self.writer.add_scalar(key, value, cur_iter)
        # write lr into tensorboard

        if isinstance(context.solver, dict):
            for k, sol in context.solver.items():
                lr = sol.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("{}_lr".format(k), lr, cur_iter)
        else:
            lr = context.solver.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("lr", lr, cur_iter)

    @classmethod
    def calcute_iteration(cls, progress):
        return (progress.epoch - 1) * progress.max_iter + progress.iter - 1


class GenerateVideoHook(BaseHook):

    def __init__(self, logdir, ckpt_dir, is_atari=False):
        self.logdir = logdir
        self.ckpt_dir = ckpt_dir
        self.is_atari = is_atari

    def after_train(self):
        trainer = self.trainer
        # Resume best model
        model = trainer.model
        ckpt = Checkpoint(os.path.join(self.ckpt_dir, "best.pkl"), model)
        filename = ckpt.get_checkpoint_file()
        logger.info("load checkpoint from {}".format(filename))
        ckpt.resume()
            
        from baserl.data.provider.collector import Collector
        from baserl.data.env import DummyVectorEnv
        import gymnasium as gym

        policy = trainer.policy
        # policy.model.eval()
        policy.eval()
        if hasattr(policy, "set_eps"):
            policy.set_eps(trainer.cfg.MODEL.TEST_EPS)
        
        if self.is_atari:
            env = wrap_deepmind(self.trainer.cfg.DATA.TASK_NAME, 
                                scale=0,
                                frame_stack=self.trainer.cfg.DATA.TEST_ENV_ARGS.get("stack_num", 1),
                                episode_life=False,
                                clip_rewards=False,
                                )
            env.seed(self.trainer.cfg.seed)
        else:
            if trainer.cfg.DATA.TASK_NAME in ["Hopper-v4", "Walker2d-v4"]:
                 env = envpool.make_gymnasium(trainer.cfg.DATA.TASK_NAME_ENVPOOL,)
            else:
                env = DummyVectorEnv(
                    [lambda: gym.make(trainer.cfg.DATA.TASK_NAME, render_mode='rgb_array'),], 
                    seed=self.trainer.cfg.seed)
        
        
        env.reset()
        
        collector = Collector(policy, 
                              env, 
                              )
        if trainer.cfg.DATA.TASK_NAME in ["Hopper-v4", "Walker2d-v4"]:
            result = collector.collect(n_episode=1)
        else:
            result = collector.collect(n_episode=1, render=0.03)
        rews, lens = result["rews"], result["lens"]
        logger.info(f"Gif reward: {rews.mean()}")

        render_imgs = result['render_imgs']
        
        with open(os.path.join(self.logdir, 'test_log'), 'w') as f:
            for i in range(len(result['log_actions'])):
                f.write(str(result['log_obs'][i][0]) + '    ' +
                    str(result['log_actions'][i][0]) + '\n')

        if trainer.cfg.DATA.TASK_NAME in ["Hopper-v4", "Walker2d-v4"]:
            pass
        else:
            save_video(render_imgs, 
                os.path.join(self.logdir, 'test_performance.gif'),
                len(render_imgs)*0.03)
            logger.info(f"saved gif to {self.logdir}")
    
    def resize_input():
        return
    


class PlotHook(BaseHook):

    def __init__(self, logdir):
        self.logdir = logdir

    def after_train(self):
        plot_dict = self.trainer.reward_meter.get_filtered_meter(filter_key="epoch_test_reward")
        plot_dict = {k:list(v._deque) for k, v in plot_dict.items()}
        plt.plot(plot_dict["epoch_test_reward_iter"], plot_dict["epoch_test_reward_mean"])
        plt.fill_between(plot_dict["epoch_test_reward_iter"], 
                         plot_dict["epoch_test_reward_min"], 
                         plot_dict["epoch_test_reward_max"], alpha=0.3)
        plt.savefig(os.path.join(self.logdir, "plot_min_max.png"))
        plt.close()
        plt.plot(plot_dict["epoch_test_reward_iter"], plot_dict["epoch_test_reward_mean"])
        plt.fill_between(plot_dict["epoch_test_reward_iter"], 
                         np.array(plot_dict["epoch_test_reward_mean"]) - np.array(plot_dict["epoch_test_reward_std"]), 
                         np.array(plot_dict["epoch_test_reward_mean"]) + np.array(plot_dict["epoch_test_reward_std"]),
                         alpha=0.3)
        plt.savefig(os.path.join(self.logdir, "plot_std.png"))
        plt.close()
        np.save(os.path.join(self.logdir, "reward_log.npy"), plot_dict)
