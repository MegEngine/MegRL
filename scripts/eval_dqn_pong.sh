#!/bin/bash

export PYTHONPATH=./
python baserl/tools/eval.py -task Atari/Pong -alg DQN -save eval_f1_SEED_1 --seed 1 --is_atari -load /data/Outputs/model_logs/train_dqn/PongNoFrameskip-v4/f1_SEED_12023-06-21_22-24-50/ckpt/

# python baserl/tools/train.py -f examples/dqn/config_eval.py