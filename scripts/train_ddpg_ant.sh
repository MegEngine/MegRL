#!/bin/bash

export PYTHONPATH=./
export MUJOCO_GL=egl
python baserl/tools/train.py -task Mujoco/Ant-v4 -alg DDPG -save f1_SEED_1 --seed 1
python baserl/tools/train.py -task Mujoco/Ant-v4 -alg DDPG -save f1_SEED_2 --seed 2
python baserl/tools/train.py -task Mujoco/Ant-v4 -alg DDPG -save f1_SEED_3 --seed 3
python baserl/tools/train.py -task Mujoco/Ant-v4 -alg DDPG -save f1_SEED_4 --seed 4
python baserl/tools/train.py -task Mujoco/Ant-v4 -alg DDPG -save f1_SEED_5 --seed 5

# python baserl/tools/train.py -f examples/ddpg/config_eval.py