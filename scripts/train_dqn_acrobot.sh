#!/bin/bash

export PYTHONPATH=./
python baserl/tools/train.py -task Classic/Acrobot-v1 -alg DQN -save f1_SEED_1_ --seed 1
# python baserl/tools/train.py -task Classic/Acrobot-v1 -alg DQN -save f1_SEED_2_ --seed 2
# python baserl/tools/train.py -task Classic/Acrobot-v1 -alg DQN -save f1_SEED_3_ --seed 3
# python baserl/tools/train.py -task Classic/Acrobot-v1 -alg DQN -save f1_SEED_4_ --seed 4
# python baserl/tools/train.py -task Classic/Acrobot-v1 -alg DQN -save f1_SEED_5_ --seed 5
# python baserl/tools/train.py -task Classic/Acrobot-v1 -alg DQN -save f1_SEED_6_ --seed 6
# python baserl/tools/train.py -task Classic/Acrobot-v1 -alg DQN -save f1_SEED_7_ --seed 7
# python baserl/tools/train.py -task Classic/Acrobot-v1 -alg DQN -save f1_SEED_8_ --seed 8
# python baserl/tools/train.py -task Classic/Acrobot-v1 -alg DQN -save f1_SEED_9_ --seed 9
# python baserl/tools/train.py -task Classic/Acrobot-v1 -alg DQN -save f1_SEED_10_ --seed 10

# python baserl/tools/train.py -f examples/dqn/config_eval.py