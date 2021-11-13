import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical
from torch.optim import lr_scheduler
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class ReplayBuffer(object):
    def __init__(self, num_total_sizes, act_dims, obs_dims, batch_size=32):
        self.num_total_sizes = num_total_sizes
        self.batch_size = batch_size
        self.observations, self.actions, self.rewards = np.zeros([num_total_sizes, obs_dims]), \
                                                        np.zeros([num_total_sizes, 1]), np.zeros([num_total_sizes, 1])
        self.old_log_probs, self.values, self.dones = np.zeros([num_total_sizes, 1]), np.zeros([num_total_sizes, 1]), \
                                                      np.zeros([num_total_sizes, 1])
        self.next_state, self.end = np.zeros([num_total_sizes, obs_dims]), np.zeros([num_total_sizes, 1])
        self.cur_index = 0

    # def store_data(self, cur_obs, cur_action, reward, done, old_log_prob, value, next_state, end):
    def store_data(self, **kwargs):
        """
        cur_obs:                   numpy.array                (obs_dims, )
        cur_action:                numpy.array                (act_dims, )
        reward:                   numpy.array                 (1,        )
        done:                     numpy.array                 (1,        )
        old_log_prob:             numpy.array                 (1,        )
        value:                    numpy.array                 (1,        )
        """
        self.observations[self.cur_index] = kwargs["state"]
        self.actions[self.cur_index] = kwargs["action"]
        self.rewards[self.cur_index] = kwargs["reward"]
        self.old_log_probs[self.cur_index] = kwargs["old_log_prob"]
        self.dones[self.cur_index] = kwargs["done"]
        self.values[self.cur_index] = kwargs["value"]
        self.next_state[self.cur_index] = kwargs["next_state"]
        self.end[self.cur_index] = kwargs["end"]
        self.cur_index += 1

    def clear_data(self):
        self.cur_index = 0

    @property
    def enough_data(self):
        return self.cur_index >= self.num_total_sizes

    @property
    def size(self):
        return self.cur_index


class ActorCritic(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size, seed, ppo_config, args):
        super(ActorCritic, self).__init__()
        self.args = args
        self.state_dim = obs_size
        self.action_dim = action_size
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        torch.manual_seed(seed)
        self._init_params(ppo_config)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    @property
    def is_on_policy(self):
        return True

    def _init_params(self, ppo_config):
        self.gamma = ppo_config["gamma"]
        self.lr = ppo_config["lr"]
        self.ppo_epoch = ppo_config["ppo_epoch"]
        self.batch_size = ppo_config["batch_size"]
        self.clip_epsilon = ppo_config["clip_epsilon"]
        self.weight_epsilon = ppo_config["weight_epsilon"]

    def clone(self, net):
        self.load_state_dict(net.state_dict())

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        x1 = self.fc3(x)
        action = F.softmax(x1, dim=1)
        value = self.fc4(x)

        return action, value

    def select_action_test(self, cur_obs_tensor):
        with torch.no_grad():
            action_probs, value = self.forward(cur_obs_tensor)
        action = np.argmax(action_probs.numpy()[0])
        # action = np.random.choice([0, 1, 2, 3], p=action_probs.numpy().ravel())
        entropy = - np.sum(action_probs.numpy()[0] * np.log(action_probs.numpy()[0]))
        log_prob = np.log(action_probs.numpy()[0][action])
        return action, log_prob, value.squeeze().detach().item(), action_probs.detach().squeeze(0).numpy(), entropy

    def get_action(self, state):
        with torch.no_grad():
            action_probs, value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy1 = dist.entropy()
        entropy = entropy1.unsqueeze(0)
        return action.item(), log_prob, action_probs.squeeze(0).numpy(), entropy.squeeze(0).item()

    def get_value(self, state):
        with torch.no_grad():
            action_probs, value = self.forward(state)
        return value.squeeze().item()

    def select_action(self, cur_obs_tensor):
        with torch.no_grad():
            action_probs, value = self.forward(cur_obs_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy1 = dist.entropy()
        entropy = entropy1.unsqueeze(0)

        return action.item(), log_prob, value.squeeze().item(), action_probs.squeeze(0).numpy(), entropy.squeeze(0).item()

    def compute_action(self, cur_obs_tensor):
        action_probs, value = self.forward(cur_obs_tensor)
        dist = Categorical(action_probs)

        entropy1 = dist.entropy()
        entropy = entropy1.unsqueeze(0)

        return dist, entropy, value

    def save_model(self, save_path, step):
        torch.save(self.state_dict(), save_path + 'ppo_model_{}.pkl'.format(step))

    def compute_before_store(self, **kwargs):
        return kwargs

    def compute_gae(self, rewards, values, dones, next_states, end, lammbda=0.95):
        gae = 0
        returns = np.zeros_like(values)
        for step in reversed(range(rewards.shape[0])):
            if end[step] or step == rewards.shape[0] - 1:
                with torch.no_grad():
                    _, _, value, _, _ = self.select_action(torch.FloatTensor(next_states[step]).unsqueeze(0))
            td_delta = rewards[step] + self.gamma * (1. - dones[step]) * value - values[step]
            value = values[step]
            gae = self.gamma * lammbda * (1. - dones[step]) * gae + td_delta
            returns[step] = gae + values[step]
        return returns

    def train_model(self, pooling):
        observations, actions = pooling.observations, pooling.actions
        rewards, dones = pooling.rewards, pooling.dones,
        values, old_log_probs = pooling.values, pooling.old_log_probs
        next_states, data_end = pooling.next_state, pooling.end
        returns = self.compute_gae(rewards, values, dones, next_states, data_end)
        advantages = returns - values

        loss, v_loss, p_loss, e_loss, ratio = self.do_training(torch.FloatTensor(returns), torch.FloatTensor(values),
                                                               torch.FloatTensor(old_log_probs), torch.FloatTensor(observations),
                                                               torch.FloatTensor(actions))

        return {"loss": loss, "v_loss": v_loss, "p_loss": p_loss, "e_loss": e_loss, "ratio": ratio, "lr": self.optimizer.state_dict()['param_groups'][0]['lr'], "advs": list(advantages.squeeze(1))}

    def do_training(self, returns, values, old_log_probs, observations, actions):
        loss_list = []
        ratio_list = []
        value_loss_list = []
        policy_loss_list = []
        entropy_loss_list = []
        batch_size = min(self.batch_size, returns.shape[0])
        for epoch in range(self.ppo_epoch):
            for idx, index in enumerate(
                    BatchSampler(SubsetRandomSampler(range(returns.shape[0])), batch_size, True)):
                # update value function
                # observations1 = observations[index][0].unsqueeze(0)
                observations1 = observations[index]
                new_dists, policy_entropys, new_values = self.compute_action(observations1)
                # print(new_log_probs.size(), policy_entropys.size(), new_values.size())
                actions1 = actions[index].squeeze(1)
                # new_log_probs = new_dists.log_prob(actions).sum(1,keepdim=True)
                new_log_probs = new_dists.log_prob(actions1).unsqueeze(1)
                # update policy function
                old_log_probs1 = old_log_probs[index]
                ratios = torch.exp(new_log_probs - old_log_probs1)
                advantages = returns[index] - values[index]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1. - self.clip_epsilon, 1. + self.clip_epsilon) * advantages
                entropy_loss = - self.weight_epsilon * policy_entropys.mean()
                policy_loss = -torch.min(surr1, surr2).mean()
                value_clipped = values[index] + (new_values - values[index]).clamp(-0.2, 0.2)
                value_clip_losses = 0.5 * (returns[index] - value_clipped).pow(2).mean()
                value_losses = 0.5 * (returns[index] - new_values).pow(2).mean()
                value_loss = torch.max(value_losses, value_clip_losses)
                # value_loss = 0.5 * (returns[index] - new_values).pow(2).mean()
                loss = value_loss + policy_loss + entropy_loss
                value_loss_list.append(value_loss.detach().item())
                policy_loss_list.append(policy_loss.detach().item())
                entropy_loss_list.append(entropy_loss.mean().detach().item())
                loss_list.append(loss.detach().item())
                if epoch == 0 and idx == 0:
                    ratio_list.append(np.mean(ratios.detach().numpy()))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return np.mean(loss_list), np.mean(value_loss_list), np.mean(policy_loss_list), np.mean(entropy_loss_list), np.mean(ratio_list)
