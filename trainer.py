import datetime
import os
import traceback
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboardX import SummaryWriter

plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class Trainer():
    def __init__(self, env, picker, picker_pooling, args, clip_epsilon, gamma, ppo_epoch, weight_epsilon):
        self.env = env
        self.picker = picker
        self.args = args
        self.picker_pooling = picker_pooling
        self.clip_epsilon, self.gamma, self.ppo_epoch, self.weight_epsilon = clip_epsilon, gamma, ppo_epoch, weight_epsilon
        self.picker_steps = 0
        self.picker_train_step = 0
        self.picker_actions = {k: deque(maxlen=30) for k in range(self.env.player_num)}
        self._create_files()

    def _create_files(self):
        self.start_time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        self.base_path = 'seed' + str(self.args.seed) + "_beta_" + str(self.args.entropy_beta) + "_s_lr_" + str(self.args.lr) + "_" + self.env.reward_setting2str() + '/'
        self.log_path = self.args.log_path
        os.makedirs(self.log_path + self.base_path, exist_ok=True)
        os.makedirs(self.args.save_path + self.base_path, exist_ok=True)
        self.error_log_file = os.path.join(self.log_path + self.base_path, "error.txt")
        self.picker_log_file = os.path.join(self.log_path + self.base_path, "picker_action.txt")
        self.step_log_file_full = os.path.join(self.log_path + self.base_path, "log_{}_full.txt".format("step"))
        self.ep_log_file = os.path.join(self.log_path + self.base_path, "log_{}.txt".format("episode"))
        for f in [self.picker_log_file, self.ep_log_file]:
            if os.path.exists(f):
                os.remove(f)
        self.writer = SummaryWriter(logdir="runs/" + self.start_time_str + "/" + self.base_path, flush_secs=30)

    def run(self):
        state_dim, action_dim = self.env.state_space_info(), self.env.action_space_info()
        while self.picker_steps <= self.args.max_steps:
            try:
                self.env.reset()
                done = self.env.done
                episode_data_list = []
                while not done:
                    state = self.env.get_state()
                    action, old_log_prob, value, prob, entropy = self.picker.select_action(torch.FloatTensor(state).unsqueeze(0))
                    self.picker_actions[self.env.step_index].append(action)
                    reward, done = self.env.step(action)
                    episode_data_list.append({"state": state,
                                              "action": action,
                                              "reward": reward,
                                              "done": True,
                                              "next_state": state,
                                              "end": True,
                                              "old_log_prob": old_log_prob,
                                              "value": value})
                    self.picker_steps += 1
                if done:
                    final_reward_list, other_dead_reward_list, pick_reward_list = self.env.refresh_final_reward_list()
                    for idx, data in enumerate(episode_data_list):
                        data["reward"] = final_reward_list[idx]
                        if not self.picker_pooling.enough_data:
                            self.picker_pooling.store_data(**data)
                        else:
                            break
                    self.train_picker()

                    # tensorboard log
                    if self.picker_steps % self.args.picker_tensorboard_inverval == 0:
                        self.log_reward("mean_final_reward", sum(final_reward_list) / float(len(final_reward_list)), self.picker_steps)
                        self.log_reward("mean_other_dead_reward", sum(other_dead_reward_list) / float(len(other_dead_reward_list)), self.picker_steps)
                        self.log_reward("pick_reward", sum(pick_reward_list) / float(len(pick_reward_list)), self.picker_steps)
                        for (name, l) in [("final_reward", final_reward_list), ("pick_reward", pick_reward_list), ("other_dead_reward", other_dead_reward_list)]:
                            for idx, r in enumerate(l):
                                self.log_reward(f"picker_{idx}_{name}", r, self.picker_steps)
                        for k, v in self.picker_actions.items():
                            self.log_policy_actions(0, np.array(v), self.picker_steps, f"action/picker_{k}")
                    self.log_picker_num(self.picker_steps)

            except Exception as e:
                with open(self.error_log_file, "a") as f:
                    f.write(traceback.format_exc())
                traceback.print_exc()
                exit()

    def train_picker(self):
        losses = {}
        if self.picker_pooling.enough_data:
            losses = self.train(self.picker, self.picker_pooling)
            self.picker_train_step += 1
            if self.picker_train_step % self.args.picker_tensorboard_inverval == 0:
                self.log_train(self.picker_train_step, "picker", losses)
            if self.picker.is_on_policy:
                self.picker_pooling.clear_data()
            if self.picker_train_step % self.args.picker_save_interval == 0:
                self.picker.save_model(self.args.save_path + self.base_path, self.picker_train_step)
        return losses

    def train(self, policy, pooling):
        return policy.train_model(pooling)

    def log_train(self, step, role, losses):
        print_labels = ["lr", "loss", "v_loss", "p_loss", "e_loss", "ratio"][:-1]
        print_str = "{} train step {}, ".format(role, step)
        for k in print_labels:
            if k in losses.keys():
                self.writer.add_scalar('{}/{}'.format(role, k), losses[k], step)
                print_str = print_str + "{}: {:.6f}, ".format(k, losses[k])
        print(f"{self.start_time_str} [{step}]: {print_str}")

    def log_policy_actions(self, index, actions, step, role):
        self.writer.add_histogram('{}/{}_actions'.format(role, index), actions, step)

    def log_reward(self, name, value, step):
        self.writer.add_scalar(f"reward/{name}", value, step)
        print(f"{self.start_time_str} [{step}]: {name}: {value}")

    def log_picker_num(self, step):
        with open(self.picker_log_file, "a") as f:
            f.write(",".join([str(k) for k in self.env.bean_num_list]) + "\n")