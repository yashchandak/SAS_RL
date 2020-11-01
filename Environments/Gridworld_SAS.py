from __future__ import print_function
import numpy as np
import matplotlib.pyplot  as plt
from matplotlib.patches import Rectangle, Circle, Arrow
from matplotlib.ticker import NullLocator
from Src.Utils.utils import Space


class Gridworld_SAS(object):
    def __init__(self,
                 action_type='discrete',  # 'discrete' {0,1} or 'continuous' [0,1]
                 n_actions=8,
                 debug=True,
                 max_step_length=0.25,
                 max_steps=20,
                 difficulty=1,
                 action_prob=0.8):

        print("difficulty", difficulty)
        self.debug = debug
        self.difficulty = difficulty
        self._n_episodes = 0

        self.action_prob = action_prob
        self.n_actions = n_actions
        self.action_space = Space(size=n_actions)
        self.observation_space = Space(low=np.zeros(2, dtype=np.float32), high=np.ones(2, dtype=np.float32), dtype=np.float32)
        self.disp_flag = False

        self.motions = self.get_action_motions(self.n_actions)

        self.wall_width = 0.05
        self.step_unit = self.wall_width - 0.005
        self.repeat = int(max_step_length / self.step_unit)

        self.max_steps = int(max_steps / max_step_length)
        self.step_reward = -0.05
        self.collision_reward = 0  # -0.05
        self.movement_reward = 0  # 1
        self.randomness = 0.2

        self.n_lidar = 0
        self.angles = np.linspace(0, 2 * np.pi, self.n_lidar + 1)[:-1]  # Get 10 lidar directions,(11th and 0th are same)
        self.lidar_angles = list(zip(np.cos(self.angles), np.sin(self.angles)))
        self.static_obstacles = self.get_static_obstacles()

        if debug:
            self.heatmap_scale = 99
            self.heatmap = np.zeros((self.heatmap_scale + 1, self.heatmap_scale + 1))

        self.reset()

    def seed(self, seed):
        self.seed = seed

    def get_embeddings(self):
        return self.motions.copy()

    def render(self):
        x, y = self.curr_pos

        # ----------------- One Time Set-up --------------------------------
        if not self.disp_flag:
            self.disp_flag = True
            # plt.axis('off')
            self.currentAxis = plt.gca()
            plt.figure(1, frameon=False) #Turns off the the boundary padding
            self.currentAxis.xaxis.set_major_locator(NullLocator()) #Turns of ticks of x axis
            self.currentAxis.yaxis.set_major_locator(NullLocator()) #Turns of ticks of y axis
            plt.ion()                                               #To avoid display blockage

            self.circle = Circle((x, y), 0.01, color='red')
            for coords in self.static_obstacles:
                x1, y1, x2, y2 = coords
                w, h = x2-x1, y2-y1
                self.currentAxis.add_patch(Rectangle((x1, y1), w, h, fill=True, color='gray'))
            print("Init done")
        # ----------------------------------------------------------------------

        for key, val in self.dynamic_obs.items():
            coords, cond = val
            if cond:
                x1, y1, x2, y2 = coords
                w, h = x2 - x1, y2 - y1
                self.objects[key] = Rectangle((x1, y1), w, h, fill=True, color='black')
                self.currentAxis.add_patch(self.objects[key])


        for key, val in self.reward_states.items():
            coords, cond = val
            if cond:
                x1, y1, x2, y2 = coords
                w, h = x2 - x1, y2 - y1
                self.objects[key] = Rectangle((x1, y1), w, h, fill=True)
                self.currentAxis.add_patch(self.objects[key])

        if len(self.angles) > 0:
            r = self.curr_state[-10:]
            coords = zip(r * np.cos(self.angles), r * np.sin(self.angles))

            for i, (w, h) in enumerate(coords):
                self.objects[str(i)] = Arrow(x, y, w, h, width=0.01, fill=True, color='lightgreen')
                self.currentAxis.add_patch(self.objects[str(i)])

        self.objects['circle'] = Circle((x, y), 0.01, color='red')
        self.currentAxis.add_patch(self.objects['circle'])

        # remove all the dynamic objects
        plt.pause(1e-7)
        for _, item in self.objects.items():
            item.remove()
        self.objects = {}

    def set_rewards(self):
        # All rewards
        self.G1_reward = + 100 #100
        self.G2_reward = 0 #- 5

    def reset(self):
        """
        Sets the environment to default conditions
        :return: None
        """
        self.set_rewards()
        self.steps_taken = 0
        self.reward_states = self.get_reward_states()
        self.dynamic_obs = self.get_dynamic_obstacles()
        self.objects = {}

        #x = 0.25
        #x = np.clip(x + np.random.randn()/30, 0.15, 0.35) # Add noise to initial x position
        self.curr_pos = np.array([0.25, 0.1])
        self.curr_state = self.make_state()

        self._n_episodes += 1

        return self.curr_state, self.get_valid_actions()

    def get_valid_actions(self):
        self.valid_actions = np.array((np.random.rand(self.n_actions) <= self.action_prob), dtype=int)
        # Make sure that there is at least one available action always.
        while not self.valid_actions.any():
            self.valid_actions = np.array((np.random.rand(self.n_actions) <= self.action_prob), dtype=int)

        return self.valid_actions

    def get_action_motions(self, n_actions):
        self.angles = np.linspace(0, 2 * np.pi, n_actions + 1)[:-1]  # Get actions equidistant radially
        motions = np.array(list(zip(np.cos(self.angles), np.sin(self.angles)))).round(3)

        # Normalize to make maximium distance covered at a step be 1
        max_dist = np.max(np.linalg.norm(motions, ord=2, axis=-1))
        motions /= max_dist

        return motions

    def step(self, action):
        assert self.valid_actions[action]

        self.steps_taken += 1
        reward = 0

        # Check if previous state was end of MDP, if it was, then we are in absorbing state currently.
        # Terminal state has a Self-loop and a 0 reward
        term = self.is_terminal()
        if term:
            return self.curr_state, 0, self.valid_actions, term, {'No INFO implemented yet'}

        motion = self.motions[action]  # Table look up for the impact/effect of the selected action
        reward += self.step_reward

        for i in range(self.repeat):
            if np.random.rand() < self.randomness:
                # Add noise some percentage of the time
                noise = np.random.rand(2)/1.415  # normalize by max L2 of noise
                delta = noise * self.step_unit  # Add noise some percentage of the time
            else:
                delta = motion * self.step_unit

            new_pos = self.curr_pos + delta  # Take a unit step in the direction of chosen action

            if self.valid_pos(new_pos):
                dist = np.linalg.norm(delta)
                reward += self.movement_reward * dist  # small reward for moving
                if dist >= self.wall_width:
                    print("ERROR: Step size bigger than wall width", new_pos, self.curr_pos, dist, delta, motion, self.step_unit)

                self.curr_pos = new_pos
                reward += self.get_goal_rewards(self.curr_pos)
            else:
                reward += self.collision_reward
                break

            # To avoid overshooting the goal
            if self.is_terminal():
                break

            # self.update_state()
            self.curr_state = self.make_state()

        if self.debug:
            # Track the positions being explored by the agent
            x_h, y_h = self.curr_pos*self.heatmap_scale
            self.heatmap[min(int(y_h), 99), min(int(x_h), 99)] += 1

            ## For visualizing obstacle crossing flaw, if any
            # for alpha in np.linspace(0,1,10):
            #     mid = alpha*prv_pos + (1-alpha)*self.curr_pos
            #     mid *= self.heatmap_scale
            #     self.heatmap[min(int(mid[1]), 99)+1, min(int(mid[0]), 99)+1] = 1

        return self.curr_state.copy(), reward, self.get_valid_actions(), self.is_terminal(), {'No INFO implemented yet'}


    def make_state(self):
        x, y = self.curr_pos
        state = [x, y]

        # Append lidar values
        for cosine, sine in self.lidar_angles:
            r, r_prv = 0, 0
            pos = (x+r*cosine, y+r*sine)
            while self.valid_pos(pos) and r < 0.5:
                r_prv = r
                r += self.step_unit
                pos = (x+r*cosine, y+r*sine)
            state.append(r_prv)

        # Append the previous action chosen
        # state.extend(self.curr_action)

        return state

    def get_goal_rewards(self, pos):
        for key, val in self.reward_states.items():
            region, reward = val
            if reward and self.in_region(pos, region):
                self.reward_states[key] = (region, 0)  # remove reward once taken
                if self.debug: print("Got reward {} in {} steps!! ".format(reward, self.steps_taken))

                return reward
        return 0

    def get_reward_states(self):
        self.G1 = (0.22, 0.47, 0.28, 0.53)
        self.G2 = (0.90, 0.05, 0.95, 0.1)
        return {'G1': (self.G1, self.G1_reward),
                'G2': (self.G2, self.G2_reward)}
    # def get_reward_states(self):
    #     self.G1 = (0.95, 0.95, 1.0, 1.0)
    #     return {'G1': (self.G1, self.G1_reward)}

    def get_dynamic_obstacles(self):
        """
        :return: dict of objects, where key = obstacle shape, val = on/off
        """
        return {}

        # self.Gate = (0.15,0.25,0.35,0.3)
        # return {'Gate': (self.Gate, self.Gate_reward)}

    def get_static_obstacles(self):
        """
        Each obstacle is a solid bar, represented by (x,y,x2,y2)
        representing bottom left and top right corners,
        in percentage relative to board size

        :return: list of objects
        """
        # gap = 0.20/self.difficulty
        #
        # # Vertical bars
        # self.O1 = (0.47, 0, 0.47 + self.wall_width, 0.17 - gap)
        # self.O2 = (0.47, 0.15 + gap, 0.47 + self.wall_width, 0.85 - gap)
        # self.O3 = (0.47, 0.85 + gap, 0.47 + self.wall_width, 1)
        #
        # # Top horizontal bars
        # self.O4 = (0, 0.65, 0.25 - gap, 0.65 + self.wall_width)
        # self.O5 = (0.25 + gap, 0.65, 0.75 - gap, 0.65 + self.wall_width)
        # self.O6 = (0.75 + gap, 0.65, 1, 0.65 + self.wall_width)
        #
        # # Bottom Horizontal bars
        # self.O7 = (0, 0.3, 0.75 - gap, 0.3 + self.wall_width)
        # self.O8 = (0.75 + gap, 0.3, 1, 0.3 + self.wall_width)
        #
        # obstacles = [self.O1, self.O2, self.O3, self.O4, self.O5, self.O6, self.O7, self.O8]
        #

        self.O1 = (0, 0.25, 0 + self.wall_width + 0.45, 0.25 + self.wall_width)  # (0, 0.25, 0.5, 0.3)
        self.O2 = (0.5, 0.25, 0.5 + self.wall_width, 0.25 + self.wall_width + 0.5)  # (0.5, 0.25, 0.55, 0.8)
        obstacles = [self.O1, self.O2]

        # obstacles = []
        return obstacles

    def valid_pos(self, pos):
        flag = True

        # Check boundary conditions
        if not self.in_region(pos, [0,0,1,1]):
            flag = False

        # Check collision with static obstacles
        for region in self.static_obstacles:
            if self.in_region(pos, region):
                flag = False
                break

        # Check collision with dynamic obstacles
        for key, val in self.dynamic_obs.items():
            region, cond = val
            if cond and self.in_region(pos, region):
                flag = False
                break

        return flag

    def is_terminal(self):
        if self.in_region(self.curr_pos, self.G1):
            return 1
        elif self.steps_taken >= self.max_steps:
            return 1
        else:
            return 0

    def in_region(self, pos, region):
        x0, y0 = pos
        x1, y1, x2, y2 = region
        if x0 >= x1 and x0 <= x2 and y0 >= y1 and y0 <= y2:
            return True
        else:
            return False


if __name__=="__main__":
    # Random Agent
    rewards_list = []
    env = Gridworld_SAS(debug=True)
    for i in range(50):
        rewards = 0
        done = False
        _, valid = env.reset()
        while not done:
            env.render()
            available = np.where(valid)[0]
            action = np.random.choice(available)
            next_state, r, valid, done, _ = env.step(action)
            rewards += r

        print(env.steps_taken)
        rewards_list.append(rewards)

    print("Average random rewards: ", np.mean(rewards_list), np.sum(rewards_list))