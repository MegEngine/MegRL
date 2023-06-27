#!/bin/bash

export PYTHONPATH=./
python baserl/tools/train.py -task Atari/Pong -alg DiscreteSAC -save f1_SEED_1 --seed 1
python baserl/tools/train.py -task Atari/Pong -alg DiscreteSAC -save f1_SEED_2 --seed 2
python baserl/tools/train.py -task Atari/Pong -alg DiscreteSAC -save f1_SEED_3 --seed 3
python baserl/tools/train.py -task Atari/Pong -alg DiscreteSAC -save f1_SEED_4 --seed 4
python baserl/tools/train.py -task Atari/Pong -alg DiscreteSAC -save f1_SEED_5 --seed 5

# python baserl/tools/train.py -f examples/dqn/config_eval.py