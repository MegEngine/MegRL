#!/bin/bash

export PYTHONPATH=./
python baserl/tools/train.py -task Classic/CartPole-v1 -alg DiscreteSAC -save f1_seed_1_ --seed 1
python baserl/tools/train.py -task Classic/CartPole-v1 -alg DiscreteSAC -save f1_seed_2_ --seed 2
python baserl/tools/train.py -task Classic/CartPole-v1 -alg DiscreteSAC -save f1_seed_3_ --seed 3
python baserl/tools/train.py -task Classic/CartPole-v1 -alg DiscreteSAC -save f1_seed_4_ --seed 4
python baserl/tools/train.py -task Classic/CartPole-v1 -alg DiscreteSAC -save f1_seed_5_ --seed 5
python baserl/tools/train.py -task Classic/CartPole-v1 -alg DiscreteSAC -save f1_seed_6_ --seed 6
python baserl/tools/train.py -task Classic/CartPole-v1 -alg DiscreteSAC -save f1_seed_7_ --seed 7
python baserl/tools/train.py -task Classic/CartPole-v1 -alg DiscreteSAC -save f1_seed_8_ --seed 8
python baserl/tools/train.py -task Classic/CartPole-v1 -alg DiscreteSAC -save f1_seed_9_ --seed 9
python baserl/tools/train.py -task Classic/CartPole-v1 -alg DiscreteSAC -save f1_seed_10_ --seed 10
# python baserl/tools/train.py -f examples/dqn/config_eval.py