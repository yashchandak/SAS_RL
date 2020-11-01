import numpy as np

class Agent:
    # Parent Class for all algorithms

    def __init__(self, config):
        self.state_low, self.state_high = config.env.observation_space.low, config.env.observation_space.high
        self.state_diff = self.state_high - self.state_low

        try:
            if config.env.action_space.dtype == np.float32:
                self.action_low, self.action_high = config.env.action_space.low, config.env.action_space.high
                self.action_diff = self.action_high - self.action_low
        except:
            print('-------------- Warning: Possible action type mismatch ---------------')

        self.state_dim = config.env.observation_space.shape[0]
        self.action_dim = config.env.action_space.shape[0]

        self.config = config
        self.entropy, self.tracker = 0, 0
        self.weights, self.grads, self.eligibility = {}, {}, {}


    def init(self):
        if self.config.restore:
            self.load()

    def save(self):
        if self.config.save_model:
            np.save(self.config.paths['ckpt'] + 'weights.npy', self.weights)
            # print("Model saved.")

    def load(self):
        try:
            self.weights = np.load(self.config.paths['ckpt'] + 'weights.npy').item()
            print('Loaded model from last checkpoint...')
        except ValueError as error:
            print("Loading failed: ", error)

    def reset(self):
        raise NotImplementedError

    def get_grads(self):
        grads = []
        if self.config.debug:
            for val in self.grads.values():
                grads.append(np.mean(np.abs(val)))
        return grads

    def track_entropy(self, act_probs, action):
        if self.config.debug:
            if self.config.cont_actions:
                # Not tracking entropy, rather just short term deviation from long term mean
                self.entropy = 0.5 * self.entropy + 0.5 * np.sum((action - self.tracker)**2)
                self.tracker = 0.99 * self.tracker + 0.01*action
            else:
                curr_entropy = - np.sum(act_probs * np.log(act_probs + 1e-8))
                self.entropy = 0.9 * self.entropy + 0.1 * curr_entropy

