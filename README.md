# MegRL

## Introduction

This is an implementation of 6 classic RL algorithms in MegEngine. The algorithms include DQN, DDPG, PG, A2C, SAC(continuous), and SAC(discrete). These algorithms can be run in the Classic Control, Atari, and MuJoCo environments in [Gymnasium](https://gymnasium.farama.org/).

## Environment

Before running the code, please install basecls[all], envpool, portalocker, h5py, numba, matplotlib, gymnasium[classic-control], gymnasium[atari], gymnasium[mujoco], AutoROM. And run:

```
AutoROM --accept-license 
```



See requirement.txt for the full environment.

##Training

You can run "python baserl/tools/train.py" to train a certain RL model on a given task, e.g.

```shell
python baserl/tools/train.py -task Classic/CartPole-v1 -alg DQN -save the_name_of_log_directory --seed 1
```



We offer many examples in the "scripts" directory. You can simply run

```sh
sh scripts/train_[algname]_[taskname].sh
```

to apply the algorithm "algname" to the task "taskname".



You can modify baserl/configs/[tasktype]\_[algname]\_cfg.py to set the hyperparameters, training scheme, log path and etc.

## Evaluation

Similar to training, you can run "python baserl/tools/eval.py" to evaluated a trained RL model on a given task, e.g.

```shell
python baserl/tools/eval.py -task Atari/Pong -alg DQN -save the_name_of_log_directory --seed 1 --is_atari -load /path/to/the/model/checkpoint
```



Please make sure the setting in baserl/configs/[tasktype]\_[algname]\_cfg.py is compatible to the trained model to be evaluated.

## Acknowledgement

The code borrows heavily from [Tianshou](https://github.com/thu-ml/tianshou) by [thu-ml](https://github.com/thu-ml), which is an RL platform based on PyTorch.

