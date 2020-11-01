import numpy as np
import itertools
from Src.Algorithms.Agent_np import Agent

def stablesoftmax(x, valid):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    exps *= valid
    return exps / np.sum(exps)

class SAS_PGQV(Agent):
    def __init__(self, config):
        super(SAS_PGQV, self).__init__(config)

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
        self.weights['v'] = np.random.randn(1, feature_dim)/np.sqrt(feature_dim)
        self.weights['q'] = np.random.randn(self.action_dim, feature_dim)/np.sqrt(feature_dim)

        self.alpha = np.array([-0.5, -0.5])  # Initialize the mixing factor as half-half
        self.rate = self.config.alpha_rate
        self.alpha_ckpts = [list(self.alpha)]

        self.idx = 0
        self.ep_rewards = np.zeros((self.config.max_steps, 1))
        self.ep_states = np.zeros((self.config.max_steps, feature_dim))
        self.ep_actions = np.zeros((self.config.max_steps, 1), dtype=int)
        self.ep_probs = np.zeros((self.config.max_steps, self.action_dim))

    def reset(self):
        self.idx = 0
        return 0

    def save(self):
        self.alpha_ckpts.append(list(self.alpha))
        np.save(self.config.paths['ckpt'] + 'alphas', self.alpha_ckpts)
        super(SAS_PGQV, self).save()

    def get_features(self, state):
        if self.Fourier_flag:
            # Normalize
            state_norm = (state - self.state_low) / self.state_diff

            # Generate fourier basis
            basis = np.dot(state_norm, self.fourier_weights.T)
            basis = np.cos(np.pi*basis)
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

        # Get the estimate of return as predicted by the two control variates.
        v_pred = np.dot(s1, self.weights['v'].T)  # Bxd x dx1 -> Bx1
        q_pred = np.sum(np.dot(s1, self.weights['q'].T) * prob1, axis=-1, keepdims=True)  # sum((Bxd x dxA)*(BxA))->Bx1

        # Td_error for the control variates
        td_error_v = G1 - v_pred  # Bx1 - Bx1 -> Bx1
        td_error_q = G1 - q_pred  # Bx1 - Bx1 -> Bx1

        # *ADD* the baselines. Let alphas control the sign appropriately.
        td_error_policy = G1 + self.alpha[0] * v_pred + self.alpha[1] * q_pred  # Bx1 - (Bx1 + Bx1 ) -> Bx1

        # derivative of softmax
        grad_score = - prob1  # BxA
        grad_score[np.arange(len(a1)), a1.flatten()] += 1   # BxA[BxA] -> BxA

        # Batch gradients. Note: It's summation over batch everywhere and not mean.
        grad_log_prob = np.einsum('ij,ik->ijk', grad_score, s1)  # BxA (x) Bxd -> BxAxd
        self.grads['v'] = np.sum(td_error_v * s1, axis=0, keepdims=True)  # sum(Bx1 * Bxd) -> sum(Bxd) -> 1xd
        self.grads['q'] = np.dot(prob1.T, td_error_q * s1)  # AxB x (Bx1 * Bxd) -> AxB x Bxd -> Axd
        self.grads['actor'] = np.sum(grad_log_prob * np.expand_dims(td_error_policy, 2), axis=0)  # sum(BxAxd * Bx1x1) -> Axd

        # Summations might increase the gradient magnitude. Therefore clip its norm.
        # self.clip_norm(max_norm=1)

        # Update the weight matrices.
        self.weights['v'] += self.config.v_lr * self.grads['v']
        self.weights['q'] += self.config.q_lr * self.grads['q']
        self.weights['actor'] += self.config.actor_lr * self.grads['actor']

        # True estimator of the gradient
        psi_G = (grad_log_prob * np.expand_dims(G1, 2)).reshape(n_samples, -1, 1)  # (BxAxd * Bx1x1).reshape()) -> Bx(Axd)x1

        # Create the two control variates independently.
        psi_v = (grad_log_prob * np.expand_dims(v_pred, 2)).reshape(n_samples, -1)  # (BxAxd * Bx1x1).reshape() -> Bx(Axd)
        psi_q = (grad_log_prob * np.expand_dims(q_pred, 2)).reshape(n_samples, -1)  # (BxAxd * Bx1x1).reshape() -> Bx(Axd)

        # Compute the covariances between the control variates and their relation with the true estimator
        psi_v_q = np.stack((psi_v, psi_q), axis=2)  # Bx(Axd) concat Bx(Axd) -> Bx(Axd)x2
        CtC = np.mean(np.einsum('ijk,ikl->ijl', psi_v_q.transpose((0, 2, 1)), psi_v_q), axis=0)  # mean(Bx2x(Axd) x Bx(Axd)x2) -> 2x2
        CtC_inv = np.linalg.inv(CtC + np.identity(2) * 1e-10)  # 2x2  # Add diagonal noise to ensure inverse exists.

        # Compute and update the alphas for next round of parameter updates.
        CtH = np.mean(np.einsum('ijk,ikl->ijl', psi_v_q.transpose((0, 2, 1)), psi_G), axis=0)  # mean(Bx2x(Axd) x Bx(Axd)x1) -> 2x1
        new_alpha = - np.dot(CtC_inv, CtH)  # 2x2 x 2x1 -> 2x1
        self.alpha = self.rate * self.alpha + (1 - self.rate) * new_alpha.flatten()  # 2 + 2 -> 2
        # self.alpha /= np.abs(np.sum(self.alpha))

        # self.alpha = np.array([0, 0])

    def clip_norm(self, max_norm=1):
        for name, param in self.grads.items():
            norm = np.linalg.norm(param, 2)
            if norm > max_norm:
                self.grads[name] = (param / norm) * max_norm


