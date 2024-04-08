import torch


class AffineInControlDynamics:
    def __init__(self, state_dim, action_dim):
        self._state_dim = state_dim
        self._action_dim = action_dim

    def f(self, x):
        raise NotImplementedError

    def g(self, x):
        raise NotImplementedError

    def mj_state_to_state(self, mj_state):
        raise NotImplementedError

    def state_to_mj_data_and_qpos(self, next_state, mj_state, data):
        raise NotImplementedError

    def rhs(self, x, action):
        action = action.unsqueeze(0) if action.ndim == 1 else action
        x = x.unsqueeze(0) if x.ndim == 1 else x
        return (self.f(x) + torch.mm(action, self.g(x).t())).squeeze_(0)
