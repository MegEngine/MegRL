#!/bin/bash

export PYTHONPATH=./
export MUJOCO_GL=egl
python baserl/tools/train.py -task Mujoco/InvertedDoublePendulum-v4 -alg A2C -save f5_seed1_ --seed 1
python baserl/tools/train.py -task Mujoco/InvertedDoublePendulum-v4 -alg A2C -save f5_seed2_ --seed 2
python baserl/tools/train.py -task Mujoco/InvertedDoublePendulum-v4 -alg A2C -save f5_seed3_ --seed 3
python baserl/tools/train.py -task Mujoco/InvertedDoublePendulum-v4 -alg A2C -save f5_seed4_ --seed 4
python baserl/tools/train.py -task Mujoco/InvertedDoublePendulum-v4 -alg A2C -save f5_seed5_ --seed 5
python baserl/tools/train.py -task Mujoco/InvertedDoublePendulum-v4 -alg A2C -save f5_seed6_ --seed 6
python baserl/tools/train.py -task Mujoco/InvertedDoublePendulum-v4 -alg A2C -save f5_seed7_ --seed 7
python baserl/tools/train.py -task Mujoco/InvertedDoublePendulum-v4 -alg A2C -save f5_seed8_ --seed 8
python baserl/tools/train.py -task Mujoco/InvertedDoublePendulum-v4 -alg A2C -save f5_seed9_ --seed 9
python baserl/tools/train.py -task Mujoco/InvertedDoublePendulum-v4 -alg A2C -save f5_seed10_ --seed 10

# python baserl/tools/train.py -f examples/pg/config_eval.py
