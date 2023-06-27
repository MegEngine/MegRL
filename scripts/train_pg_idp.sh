#!/bin/bash

export PYTHONPATH=./
export MUJOCO_GL=egl
python baserl/tools/train.py -task Mujoco/InvertedDoublePendulum-v4 -alg PG -save f1_seed1_ --seed 1
python baserl/tools/train.py -task Mujoco/InvertedDoublePendulum-v4 -alg PG -save f1_seed2_ --seed 2
python baserl/tools/train.py -task Mujoco/InvertedDoublePendulum-v4 -alg PG -save f1_seed3_ --seed 3
python baserl/tools/train.py -task Mujoco/InvertedDoublePendulum-v4 -alg PG -save f1_seed4_ --seed 4
python baserl/tools/train.py -task Mujoco/InvertedDoublePendulum-v4 -alg PG -save f1_seed5_ --seed 5

# python baserl/tools/train.py -f examples/pg/config_eval.py
