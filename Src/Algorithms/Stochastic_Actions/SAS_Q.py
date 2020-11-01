import numpy as np
import itertools
from Src.Algorithms.Agent_np import Agent

class SAS_Q(Agent):
    def __init__(self, config):
        super(SAS_Q, self).__init__(config)

        self.Fourier_flag = False
        if self.state_dim > 3:
            print("Wartning: Fourier cannot be used. Using Linear...")
            feature_dim = self.state_dim
        else:
            fourier_order = config.fourier_order
            self.fourier_weights = np.array(list(itertools.product(np.arange(0, fourier_order + 1), repeat=self.state_dim)))
            feature_dim = self.fourier_weights.shape[0]
            self.Fourier_flag = True

        self.weights['q'] = np.random.randn(self.action_dim, feature_dim)/np.sqrt(feature_dim)
        self.buffer = MemoryBuffer(max_len=self.config.buffer_size, state_dim=feature_dim,
                                   action_dim=1, atype=np.int,
                                   valid_dim=self.action_dim, stype=np.float32)
        self.alpha = [0,0]

    def reset(self):
        return 0

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
        if np.random.rand() > self.config.exp:
            q_vals = np.dot(feats, self.weights['q'].T)
            q_vals[np.logical_not(valid)] = - np.inf
            action = np.argmax(q_vals)
        else:
            action = np.random.choice(np.where(valid)[0])

        return action, feats

    def update(self, s1, a1, feats, r1, s2, valid, done):
        feats2 = self.get_features(s2)
        self.buffer.add(feats, a1, r1, feats2, valid, int(done != 1))

        if done and self.buffer.length > self.config.batch_size:
            self.optimize()

    def optimize(self):
        for i in range(self.config.SAS_q_updates):
            s1, a1, r1, s2, valid, not_absorbing = self.buffer.sample(self.config.batch_size)

            # Create a binary matrix (BxA) to store the actions taken
            action_matrix = np.zeros(valid.shape, dtype=bool)  # Note: Using int causes a bug in indexing grad_score
            action_matrix[np.arange(len(a1)), a1.flatten()] = True

            q_pred = np.dot(s1, self.weights['q'].T)  # Bxd x dxA -> BxA
            q_pred = np.expand_dims(q_pred[action_matrix], axis=1)  # BxA[BxA].expand() -> Bx1

            q_target = np.dot(s2, self.weights['q'].T)  # Bxd x dxA -> BxA
            q_target[np.logical_not(valid)] = - np.inf
            q_max = np.max(q_target, axis=-1, keepdims=True)  # Bx1

            td_error = r1 + self.config.gamma * q_max * not_absorbing - q_pred  # Bx1 + Bx1 * Bx1 - Bx1 -> Bx1

            # Batch gradients. Note: It's summation over batch. Hence divide by batch size.
            temp = td_error * s1  # Bx1 * Bxd -> Bxd
            self.grads['q'] = np.dot(action_matrix.T, temp)/self.config.batch_size  # AxB x Bxd -> Axd

            # Update the weight matrices.
            self.weights['q'] += self.config.q_lr * self.grads['q']


class MemoryBuffer:
    """
    Pre-allocated memory interface for storing and using Off-policy trajectories
   """
    def __init__(self, max_len, state_dim, action_dim, atype, valid_dim=1, stype=np.float32):
        self.s1 = np.zeros((max_len, state_dim), dtype=stype)
        self.a1 = np.zeros((max_len, action_dim), dtype=atype)
        self.r1 = np.zeros((max_len, 1), dtype=np.float32)
        self.s2 = np.zeros((max_len, state_dim), dtype=stype)
        self.valid = np.zeros((max_len, valid_dim), dtype=bool)
        self.done = np.zeros((max_len, 1), dtype=np.float32)

        self.length = 0
        self.max_len = max_len
        self.atype = atype
        self.stype = stype

    @property
    def size(self):
        return self.length

    def reset(self):
        self.length = 0

    def _get(self, ids):
        return self.s1[ids], self.a1[ids], self.r1[ids], self.s2[ids], self.valid[ids], self.done[ids]

    def sample(self, batch_size):
        count = min(batch_size, self.length)
        return self._get(np.random.choice(self.length, count))

    def add(self, s1, a1, r1, s2, valid, done):
        pos = self.length
        if self.length < self.max_len:
            self.length = self.length + 1
        else:
            pos = np.random.randint(self.max_len)

        self.s1[pos] = s1
        self.a1[pos] = a1
        self.r1[pos] = r1
        self.s2[pos] = s2
        self.valid[pos] = valid
        self.done[pos] = done



