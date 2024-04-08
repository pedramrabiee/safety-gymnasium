from safety_gymnasium.bases.base_dynamics import AffineInControlDynamics
import torch
from scipy.spatial.transform import Rotation as R
import numpy as np

class UnicycleDynamics(AffineInControlDynamics):
    # states:
    # def f(self, x):
    #     return torch.tensor([x[..., 2] * torch.cos(x[..., 3]), x[..., 2] * torch.sin(x[..., 3]), 0., 0.]).unsqueeze_(-1)
    def f(self, x):
        return torch.tensor([x[..., 2] * torch.cos(x[..., 3]), x[..., 2] * torch.sin(x[..., 3]), 0., 0.])

    def g(self, x):
        return torch.vstack([torch.zeros(2, 2, dtype=torch.float64), torch.eye(2, dtype=torch.float64)])

    def mj_state_to_state(self, mj_state):
        pos, vel, mat = mj_state['pos'], mj_state['vel'], mj_state['mat']
        angle =  R.from_matrix(mat.reshape(3, -1)).as_euler('xyz')[-1]
        return np.array([pos[0], pos[1], np.linalg.norm([vel[:2]]), angle])
    def next_state_to_mj_data_and_qpos(self, next_state, mj_state, data):
        angle0 = R.from_matrix(mj_state['mat0'].reshape(3, -1)).as_euler('xyz')[-1]
        data.qpos[0] += np.cos(angle0) * (next_state[0] - mj_state['pos'][0])
        data.qpos[1] += np.sin(-angle0) * (next_state[1] - mj_state['pos'][1])

        mj_state['pos'][:2] = next_state[:2]

        prev_angle = R.from_matrix(mj_state['mat'].reshape(3, -1)).as_euler('xyz')[-1]
        mj_state['mat'] = R.from_euler('xyz', np.array([0.0, 0.0, next_state[3]])).as_matrix().flatten()
        mj_state['vel'][0] = np.cos(next_state[3]) * next_state[2]
        mj_state['vel'][1] = np.sin(next_state[3]) * next_state[2]
        # data.qpos[2] += next_state[3] - prev_angle
        data.qpos[2] = next_state[3]
        return mj_state








