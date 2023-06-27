#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import importlib
import multiprocessing as mp
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import sys
import megfile
from loguru import logger

import megengine as mge
import megengine.distributed as dist
from megengine.device import get_device_count

import baserl.configs as BC
from baserl.utils import setup_logger, all_register, Checkpoint, save_video
from baserl.data.env.wrapper import wrap_deepmind

import envpool
from basecore.utils import str_timestamp

import random
import numpy as np

def default_parser():
    parser = argparse.ArgumentParser(description="A script that train model")
    parser.add_argument(
        "-task", "--task", default="cartpole", type=str, help="taskname"
    )
    parser.add_argument(
        "-alg", "--algname", default="dqn", type=str, help="algname[DQN, DDPG, A2C, PPO, PG, ContinuousSAC, DiscreteSac]"
    )
    parser.add_argument(
        "-d", "--dir", default=None, type=str, help="training process description file dir"
    )
    parser.add_argument(
        "-w", "--weight_file", default=None, type=str, help="weights file",
    )
    parser.add_argument(
        "-n", "--ngpus", default=None, type=int, help="total number of gpus for training",
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume training from saved checkpoint or not",
    )
    parser.add_argument(
        "--tensorboard", "--tb", action="store_true", help="use tensorboard or not",
    )
    parser.add_argument(
        "--amp", action="store_true", help="use amp during training or not",
    )
    parser.add_argument(
        "--ema", action="store_true", help="use model ema during training or not",
    )
    parser.add_argument(
        "--dtr", action="store_true",
        help="use dtr during training or not, enable while GPU memory is not enough",
    )
    parser.add_argument(
        "--sync-level", type=int, default=None, help="config sync level, use 0 to debug"
    )
    parser.add_argument(
        "--mp-method", type=str, default="fork", help="mp start method, use fork by defalut"
    )
    parser.add_argument("--fastrun", action="store_true")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="random seed"
    )
    parser.add_argument(
        "-save", "--save_name", default="", type=str, help="name of the log subdir"
    )
    parser.add_argument(
        "--is_atari", default=False, help="is atari env or not", action="store_true"
    )
    parser.add_argument(
        "-load", "--load_dir", default="", type=str, help="dirname of the model to be evaluated"
    )
    return parser


def launch_workers(args):
    rank = dist.get_rank()
    logger.info("Init process group for gpu{} done".format(rank))
    print(rank)
    
    args.seed += rank * 1000 # otherwise all process will have the same data 

    task_type, task_name = args.task.split("/")
    current_network = eval(f"BC.{task_type}{args.algname}Config")
    print(f"{task_name}-{args.algname}")
    cfg = current_network(task_name)
    cfg.merge(args.opts)

    if args.weight_file is not None:
        cfg.MODEL.WEIGHTS = args.weight_file
    if args.resume:
        cfg.TRAINER.RESUME = True
    if args.amp:
        cfg.TRAINER.AMP.ENABLE = True
    if args.ema:
        cfg.TRAINER.EMA.ENABLE = True
    if args.tensorboard:
        cfg.GLOBAL.TENSORBOARD.ENABLE = True

    cfg.GLOBAL.OUTPUT_DIR = os.path.join(cfg.GLOBAL.OUTPUT_DIR, args.save_name+args.time_str)
    cfg.GLOBAL.CKPT_SAVE_DIR = os.path.join(cfg.GLOBAL.CKPT_SAVE_DIR, args.save_name+args.time_str)
    setup_logger(log_path=cfg.GLOBAL.OUTPUT_DIR, 
                 to_loguru=True)
    logger.info("args: " + str(args))

    if args.fastrun:
        logger.info("Using fastrun mode...")
        mge.functional.debug_param.set_execution_strategy("PROFILE")

    if args.dtr:
        logger.info("Using megengine DTR")
        mge.dtr.enable()

    if args.sync_level is not None:
        # NOTE: use sync_level = 0 to debug mge error
        from megengine.core._imperative_rt.core2 import config_async_level
        logger.info("Using aysnc_level {}".format(args.sync_level))
        config_async_level(args.sync_level)

    cfg.seed = args.seed
    trainer = cfg.build_trainer()
    del cfg
    # trainer.train()
    
    
    model = trainer.model
    ckpt = Checkpoint(os.path.join(args.load_dir, "best.pkl"), model)
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
    
    if args.is_atari:
        env = wrap_deepmind(trainer.cfg.DATA.TASK_NAME, 
                            scale=0,
                            frame_stack=trainer.cfg.DATA.TEST_ENV_ARGS.get("stack_num", 1),
                            episode_life=False,
                            clip_rewards=False,
                            )
        env.seed(trainer.cfg.seed)
    else:
        if trainer.cfg.DATA.TASK_NAME in ["Hopper-v4", "Walker2d-v4"]:
                env = envpool.make_gymnasium(trainer.DATA.TASK_NAME_ENVPOOL,)
        else:
            env = DummyVectorEnv(
                [lambda: gym.make(trainer.cfg.DATA.TASK_NAME, render_mode='rgb_array'),], 
                seed=trainer.cfg.seed)
    
    env.reset()
    
    collector = Collector(policy, 
                            env, 
                            )
    result = collector.collect(n_episode=1, render=0.03)
    rews, lens = result["rews"], result["lens"]
    logger.info(f"Gif reward: {rews.mean()}")

    render_imgs = result['render_imgs']
    
    with open(os.path.join(trainer.cfg.GLOBAL.OUTPUT_DIR, 'test_log'), 'w') as f:
        for i in range(len(result['log_actions'])):
            f.write(str(result['log_obs'][i][0]) + '    ' +
                str(result['log_actions'][i][0]) + '\n')

    if trainer.cfg.DATA.TASK_NAME in ["Hopper-v4", "Walker2d-v4"]:
        pass
    else:
        save_video(render_imgs, 
            os.path.join(trainer.cfg.GLOBAL.OUTPUT_DIR, 'test_performance.gif'),
            len(render_imgs)*0.03)
        logger.info(f"saved gif to {trainer.cfg.GLOBAL.OUTPUT_DIR}")
    
    
    env = envpool.make_gymnasium(
        trainer.cfg.DATA.TASK_NAME_ENVPOOL,
        num_envs=trainer.cfg.DATA.TEST_ENV_NUM,
        seed=trainer.cfg.seed,
        **trainer.cfg.DATA.TEST_ENV_ARGS,
    )
    
    collector = Collector(policy, env, exploration_noise=trainer.cfg.MODEL.get("TEST_EXPLORATION_NOISE", False))
    result = collector.collect(n_episode=trainer.cfg.DATA.TEST_ENV_NUM, render=0.)
    logger.info(f"Evaluation Return: {result['rew']}±{result['rew_std']}")
    
    logger.info(f"Process {dist.get_rank()} finished!")


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    mge.random.seed(seed)
    mge.config.deterministic_kernel = True
    
    # mge.config.async_level = 0 # for debug


@logger.catch
def main():
    all_register()
    parser = default_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    assert args.mp_method in ["fork", "spawn", "forkserver"]
    mp.set_start_method(method=args.mp_method)

    if args.ngpus is None:
        num_devices = get_device_count("gpu")
    elif args.ngpus < 0:
        raise ValueError(f"negative device number: {args.ngpus}")
    else:
        num_devices = args.ngpus
    
    args.time_str = str_timestamp()

    def run():
        # print(num_devices)
        if num_devices > 1:
            train = dist.launcher(launch_workers, n_gpus=num_devices)
            train(args)
            # print("-------------------------")
        else:
            launch_workers(args)

    run()


if __name__ == "__main__":
    main()
