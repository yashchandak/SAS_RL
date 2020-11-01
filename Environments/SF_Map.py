from __future__ import print_function
import numpy as np
from Src.Utils.utils import Space

#TODO: add geodesic cost to each of the edge

adjacency = {
             0: [1, 26],
             1: [0, 2, 3],
             2: [1, 3, 4],
             3: [1, 2, 5],
             4: [2, 4, 6],
             5: [3, 4, 7],
             6: [4, 7, 8],
             7: [5, 6, 9, 15],
             8: [6, 9, 10],
             9: [7, 8, 11, 12],
             10: [8, 11, 13],
             11: [9, 10, 12, 14],
             12: [9, 11, 15],
             13: [10, 14, 18],
             14: [11, 13, 17],
             15: [7, 12, 16],
             16: [15, 17, 21],
             17: [14, 16, 18, 20],
             18: [13, 17, 19],
             19: [18, 20],
             20: [17, 19, 21, 22, 23],
             21: [5, 16, 20, 22],
             22: [20, 21, 24],
             23: [20, 24, 26],
             24: [23, 25],
             25: [22, 24, 27],
             26: [23, 27],
             27: [0, 25, 26, 27]
            }


class SF_Map(object):
    def __init__(self,
                 action_type='discrete',  # 'discrete' {0,1} or 'continuous' [0,1]
                 debug=False,
                 max_steps=30,
                 action_prob=0.5,
                 bridge_prob=-1):

        self.debug = debug

        self.n_states = len(adjacency.keys())
        self.n_max_action = max([len(v) for v in adjacency.values()])

        self.action_space = Space(size=self.n_max_action)
        self.observation_space = Space(size=self.n_states)

        self.max_steps = max_steps
        self.action_prob = action_prob
        self.bridge_prob = action_prob if bridge_prob < 0 else bridge_prob
        self.all_action_probs = self._get_probs()
        self.step_reward = -0.1
        self.self_loop_reward = -1
        self.goal_reward = 10

        # self.start_state = np.random.randint(self.n_states)
        self.goal_state = 0

        self.reset()

    def _get_probs(self):
        probs = {}
        for k, v in adjacency.items():
            probs[k] = [self.action_prob for _ in v]
            probs[k].extend([0]*(self.n_max_action - len(v)))  # Add zero prob for additional edges

        probs[27][0] = self.bridge_prob
        return probs

    def seed(self, seed):
        self.seed = seed

    def reset(self):
        """
        Sets the environment to default conditions
        :return: state, valid actions
        """
        self.steps_taken = 0
        self.curr_state = np.random.randint(1, self.n_states) #self.start_state

        return self._state_encoding(), self._get_valid_actions()

    def _state_encoding(self):
        # Create one hot encoding for the current state
        state_one_hot = np.zeros(self.n_states)
        state_one_hot[self.curr_state] = 1

        return state_one_hot

    def _get_valid_actions(self):
        probs = self.all_action_probs[self.curr_state]

        self.valid_actions = np.array((np.random.rand(len(probs)) <= probs), dtype=int)
        # Make sure that there is at least one available action always.
        while not self.valid_actions.any():
            self.valid_actions = np.array((np.random.rand(len(probs)) <= probs), dtype=int)

        return self.valid_actions

    def step(self, action):
        assert self.valid_actions[action]

        self.steps_taken += 1
        terminal_flag = self.steps_taken >= self.max_steps
        reward = 0

        # Check if previous state was end of MDP, if it was, then we are in absorbing state currently.
        # Terminal state has a Self-loop and a 0 reward
        if self.curr_state == self.goal_state:
            return self._state_encoding(), 0, True, {'No INFO implemented yet'}

        # Encourage shortest path by penalize for each step
        reward += self.step_reward

        # penalize for taking the self-loop at bridge
        if self.curr_state == adjacency[self.curr_state][action]:
            reward += self.self_loop_reward

        self.curr_state = adjacency[self.curr_state][action]

        if self.curr_state == self.goal_state:
            reward += self.goal_reward
            terminal_flag = True
            if self.debug:
                print("Goal reached in {}".format(self.steps_taken))

        return self._state_encoding(), reward, self._get_valid_actions(), terminal_flag, {'No INFO implemented yet'}


if __name__=="__main__":
    # Random Agent
    rewards_list = []
    env = SF_Map(debug=True)
    for i in range(50):
        rewards = 0
        done = False
        _, valid = env.reset()
        while not done:
            available = np.where(valid)[0]
            action = np.random.choice(available)
            next_state, r, valid, done, _ = env.step(action)
            rewards += r
        rewards_list.append(rewards)

    print("Average random rewards: ", np.mean(rewards_list), np.sum(rewards_list))