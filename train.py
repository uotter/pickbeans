import argparse
import os
import random

import numpy as np
import torch

from discrete_ac import ActorCritic, ReplayBuffer
from env.pickbeans import PickBeansGame
from trainer import Trainer


def training_process(args):
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    reward_config = {
        "dead": -10,
        "alive": 10,
        "other_dead": 0
    }
    env = PickBeansGame(reward_setting=reward_config)
    obs_dims, act_dims = env.state_space_info(), env.action_space_info()
    picker_buffer_config = {
        'num_total_sizes': args.picker_buff_size
    }
    ppo_config = {
        'ppo_epoch': args.ppo_epoch,
        'clip_epsilon': 0.2,
        'gamma': args.gamma,
        'weight_epsilon': args.entropy_beta,
        'lr': args.lr,
        'batch_size': 32,
    }

    picker_sample_pooling = ReplayBuffer(num_total_sizes=picker_buffer_config['num_total_sizes'], act_dims=act_dims, obs_dims=obs_dims)
    picker = ActorCritic(obs_size=obs_dims, action_size=act_dims, hidden_size=128, seed=args.seed, ppo_config=ppo_config, args=args)

    trainer = Trainer(env, picker, picker_sample_pooling, clip_epsilon=ppo_config['clip_epsilon'], gamma=ppo_config['gamma'],
                      ppo_epoch=ppo_config['ppo_epoch'], weight_epsilon=ppo_config['weight_epsilon'], args=args)
    trainer.run()


if __name__ == '__main__':
    def args_parse():
        parser = argparse.ArgumentParser()
        parser.add_argument('--save_path', type=str, default='./zlog_ppo/models/', help='Path to save a model during training.')
        parser.add_argument('--picker_save_interval', type=int, default=10, help='save step')
        parser.add_argument('--log_path', type=str, default='./zlog_ppo/logdir/', help='Path to save test result.')
        parser.add_argument('--picker_buff_size', type=int, default=128, help="the pooling buffer size of picker")
        parser.add_argument('--seed', type=int, default=7417, help='random seed')
        parser.add_argument('--max_steps', type=int, default=3e7, help='Number of total steps')
        parser.add_argument('--ppo_epoch', type=int, default=10, help='Number of training times for each minibatch in ppo')
        parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        parser.add_argument('--entropy_beta', type=float, default=0.1, help='Entropy beta in policy loss')
        parser.add_argument('--gamma', type=float, default=0.99, help='ed factor')
        parser.add_argument('--gae', type=float, default=0.97, help='gae')
        parser.add_argument('--picker_tensorboard_inverval', type=int, default=5000, help="log picker's action,  teacher's reward and loss")
        parser.add_argument('--picker_test_interval', type=int, default=1000, help="test frequency of picker")
        parser.add_argument('--picker_test_count', type=int, default=10, help="number of test episode for each test")
        args = parser.parse_args()
        return args


    training_process(args_parse())
