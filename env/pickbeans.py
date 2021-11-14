# -*- coding: utf-8 -*-
import random


class PickBeansGame():
    def __init__(self, player_num=5, bean_num=50, debug=False, **kwargs):
        self.debug = debug
        self.player_num = player_num
        self.total_bean_num = bean_num
        self.left_bean_num = bean_num
        self.reward_setting = kwargs.get("reward_setting", {
            "dead": -10,
            "alive": 10,
            "other_dead": 5
        })
        self.players = self._init_players()

    def reward_setting2str(self):
        return "_".join([f"{k}_{v}" for k, v in self.reward_setting.items()])

    def _init_players(self):
        return [Player(k) for k in range(self.player_num)]

    def _reset(self):
        self.step_index = 0
        self.final_reward_list = [0] * self.player_num
        self.bean_num_list = [0] * self.player_num
        self.state = self.compute_state(self.step_index)
        self.reward = 0
        self.done = False
        self.left_bean_num = self.total_bean_num

    def compute_state(self, step_index):
        step_index_onehot = [0.] * self.player_num
        step_index_onehot[step_index] = 1.
        return step_index_onehot + [self.left_bean_num / float(self.total_bean_num)]

    def reset(self):
        """
        reset game
        Returns:
             state, 0, done, state
        """
        self._reset()

    def state_space_info(self):
        return self.player_num + 1

    def action_space_info(self):
        return self.total_bean_num + 1

    def get_state(self):
        """
        :return current state
        """
        return self.state

    def step(self, action):
        """
        :param action
        :return reward, done
        """
        current_choose_bean_num = int((action / self.total_bean_num) * self.left_bean_num)
        # current_choose_bean_num = action if (self.left_bean_num - action) >= 0 else self.left_bean_num
        self.bean_num_list[self.step_index] = current_choose_bean_num
        self.left_bean_num = self.left_bean_num - current_choose_bean_num
        self.step_index += 1
        if self.step_index == self.player_num:
            self.done = True
        else:
            self.done = False
            self.state = self.compute_state(self.step_index)
        return self.reward, self.done

    def refresh_final_reward_list(self):
        assert sum(self.bean_num_list) <= self.total_bean_num, f"Error: total number of beans selected {sum(self.bean_num_list)} is bigger than the total bean number {self.total_bean_num}."
        max_bean_num, min_bean_num = max(self.bean_num_list), min(self.bean_num_list)
        dead_player_num = 0
        for idx, choose_bean_num in enumerate(self.bean_num_list):
            if choose_bean_num in [max_bean_num, min_bean_num]:
                if self.debug:
                    print(f"player {idx} choose {choose_bean_num} beans and in the min/max number of {(min_bean_num, max_bean_num)}, dead.")
                self.final_reward_list[idx] = self.reward_setting["dead"]
                dead_player_num += 1
            else:
                self.final_reward_list[idx] = self.reward_setting["alive"]
        if self.debug:
            print(f"reward list after computing self alive and dead: {self.final_reward_list}")
        pick_reward = [k for k in self.final_reward_list]
        # others' death will benefit self
        other_dead_reward_list = [0] * self.player_num
        for idx, dead_player_reward in enumerate(self.final_reward_list):
            other_dead_reward_list[idx] = max((dead_player_num - 2), 0) * self.reward_setting["other_dead"] if dead_player_reward == self.reward_setting["dead"] else 0
        if self.debug:
            print(f"others' death reward list: {other_dead_reward_list}")
        self.final_reward_list = [a + b for (a, b) in zip(self.final_reward_list, other_dead_reward_list)]
        if self.debug:
            print(f"reward list after computing others' alive and dead: {self.final_reward_list}")
        return self.final_reward_list, other_dead_reward_list, pick_reward

    def run(self):
        done = False
        state_dim, action_dim = self.state_space_info(), self.action_space_info()
        print(f"state size: {state_dim}, action  size: {action_dim}.")
        print('-----------')
        print(self.reset())
        print('-----------')
        while not done:
            state = self.get_state()
            action = random.randint(0, action_dim - 1)
            reward, done = self.step(action)
            print(f"player index: {self.step_index}, state: {state}, action: {action}, done: {self.done}, next state: {self.get_state()}, left beans: {self.left_bean_num}.")
            print(f"bean number list: {self.bean_num_list}, sum {sum(self.bean_num_list)}")
            print('-----------')
        if done:
            final_reward_list, other_dead_reward_list, pick_reward = self.refresh_final_reward_list()
            print(f"final reward list: {final_reward_list}")


class Player():
    def __init__(self, idx):
        self.idx = idx


if __name__ == "__main__":
    test_game = PickBeansGame(debug=True)
    test_game.run()
