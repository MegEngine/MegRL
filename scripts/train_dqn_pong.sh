#!/bin/bash

export PYTHONPATH=./
python baserl/tools/train.py -task Atari/Pong -alg DQN -save f1_SEED_1 --seed 1
python baserl/tools/train.py -task Atari/Pong -alg DQN -save f1_SEED_2 --seed 2
python baserl/tools/train.py -task Atari/Pong -alg DQN -save f1_SEED_3 --seed 3
python baserl/tools/train.py -task Atari/Pong -alg DQN -save f1_SEED_4 --seed 4
python baserl/tools/train.py -task Atari/Pong -alg DQN -save f1_SEED_5 --seed 5
python baserl/tools/train.py -task Atari/Pong -alg DQN -save f1_SEED_6 --seed 6
python baserl/tools/train.py -task Atari/Pong -alg DQN -save f1_SEED_7 --seed 7
python baserl/tools/train.py -task Atari/Pong -alg DQN -save f1_SEED_8 --seed 8
python baserl/tools/train.py -task Atari/Pong -alg DQN -save f1_SEED_9 --seed 9
python baserl/tools/train.py -task Atari/Pong -alg DQN -save f1_SEED_10 --seed 10

# python baserl/tools/train.py -f examples/dqn/config_eval.py