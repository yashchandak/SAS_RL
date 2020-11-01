import numpy as np
import itertools
from Src.Algorithms.Agent_np import Agent

def stablesoftmax(x, valid):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    exps *= valid
    return exps / np.sum(exps)

class SAS_NAC(Agent):
    def __init__(self, config):
        super(SAS_NAC, self).__init__(config)

        self.Fourier_flag = False
        if self.state_dim > 3:
            print("Wartning: Fourier cannot be used. Using Linear...")
            feature_dim = self.state_dim
        else:
            fourier_order = config.fourier_order
            self.fourier_weights = np.array(list(itertools.product(np.arange(0, fourier_order + 1), repeat=self.state_dim)))
            feature_dim = self.fourier_weights.shape[0]
            self.Fourier_flag = True

        self.weights['actor'] = np.random.randn(self.action_dim, feature_dim)/np.sqrt(feature_dim)
        self.weights['q'] = np.random.randn(self.action_dim, feature_dim)/np.sqrt(feature_dim)
        self.alpha = [0, 1]  # unused variables. Needed for the runner to track it for all algorithms.

        self.idx = 0
        self.ep_rewards = np.zeros((self.config.max_steps, 1))
        self.ep_states = np.zeros((self.config.max_steps, feature_dim))
        self.ep_actions = np.zeros((self.config.max_steps, 1), dtype=int)
        self.ep_probs = np.zeros((self.config.max_steps, self.action_dim))

    def reset(self):
        self.idx = 0
        return 0

    def save(self):
        super(SAS_NAC, self).save()

    def get_features(self, state):
        if self.Fourier_flag:
            # Normalize
            state_norm = (state - self.state_low) / self.state_diff

            # Generate fourier basis
            basis = np.dot(state_norm, self.fourier_weights.T)
            basis = np.cos(np.pi * basis)
            return basis

        else:
            return state

    def get_action(self, state, valid, explore=0):
        feats = self.get_features(state)
        score = np.dot(feats, self.weights['actor'].T)
        prob = stablesoftmax(score, valid)
        action = np.random.choice(self.action_dim, p=prob)

        # self.track_entropy(prob, action)
        return action, (prob, feats)

    def update(self, s1, a1, prob, r1, s2, valid, done):
        prob1, feats = prob

        self.ep_rewards[self.idx] = r1
        self.ep_states[self.idx] = feats
        self.ep_actions[self.idx] = a1
        self.ep_probs[self.idx] = prob1
        self.idx += 1

        if done:
            # Compute gamma return and do on-policy update
            G = 0
            for i in range(self.idx-1, -1, -1):
                r = self.ep_rewards[i][0]
                G = r + self.config.gamma * G
                self.ep_rewards[i] = G

            self.optimize(self.ep_states[:self.idx], self.ep_actions[:self.idx], self.ep_rewards[:self.idx], self.ep_probs[:self.idx])

            # Reset the episode history
            self.idx = 0

    def optimize(self, s1, a1, G1, prob1):
        n_samples = s1.shape[0]  # Batch size: B

        # derivative of softmax
        grad_score = - prob1  # BxA
        grad_score[np.arange(len(a1)), a1.flatten()] += 1   # BxA[BxA] -> BxA

        # Batch gradients. Note: It's summation over batch everywhere and not mean.
        grad_log_prob = np.einsum('ij,ik->ijk', grad_score, s1)  # BxA (x) Bxd -> BxAxd

        # Get the estimate of return using compatibility features
        q_pred = np.sum(grad_log_prob * self.weights['q'], axis=(1, 2), keepdims=True)  # sum(BxAxd * A*d) -> Bx1x1

        # Td_error
        td_error = np.expand_dims(G1, 2) - q_pred  # expand(Bx1) - Bx1x1 -> Bx1x1

        # Compute the weights for return predictor
        self.grads['q'] = np.sum(td_error * grad_log_prob, axis=0)  # sum(Bx1x1 * BxAxd) -> sum(BxAxd) -> Axd

        # Summations might increase the gradient magnitude. Therefore clip its norm.
        # self.clip_norm(max_norm=1)

        # Update the weight of the predictor
        self.weights['q'] += self.config.q_lr * self.grads['q']

        # Update the actor parameters in the natural gradient direction using the weights of return predictor.
        self.weights['actor'] += self.config.actor_lr * self.weights['q'] / np.linalg.norm(self.weights['q'], 2)


    def clip_norm(self, max_norm=1):
        for name, param in self.grads.items():
            norm = np.linalg.norm(param, 2)
            if norm > max_norm:
                self.grads[name] = (param / norm) * max_norm


