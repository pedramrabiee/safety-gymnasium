from safety_gymnasium.bases.base_task import BaseTask
from safety_gymnasium.utils.common_utils import MujocoException
import numpy as np
import mujoco
from torchdiffeq import odeint
from functools import partial
import torch

class SimulationOverrideTask(BaseTask):
    def reset(self) -> None:
        super().reset()
        self.state = dict(pos=self.agent.pos,
                          vel=self.agent.vel,
                          mat=self.agent.mat,
                          mat0=self.agent.mat)

        self.timestep = float(self.world.xml['mujoco']['option']['@timestep'])
    def simulation_forward(self, action: np.ndarray) -> None:
        """Take a step in the physics simulation by manually integrating the agent's dynamics

        Note:
            - The **step** mentioned above is not the same as the **step** in Mujoco sense.
            - The **step** here is the step in episode sense.
        """
        # Simulate physics forward
        # if self.debug:
        #     self.agent.debug()
        exception = False
        try:
            for mocap in self._mocaps.values():
                mocap.move()
            # pylint: disable-next=no-member
            self._integrate_forward(action)
        except MujocoException as me:  # pylint: disable=invalid-name
            print('MujocoException', me)
            exception = True

        if exception:
            return exception

        # pylint: disable-next=no-member
        mujoco.mj_forward(self.model, self.data)  # Needed to get sensor readings correct!
        return exception

    def _integrate_forward(self, action):
        state = self.agent.dynamics.mj_state_to_state(self.state)
        next_state = odeint(func=lambda t, y: partial(self.agent.dynamics.rhs, action=torch.from_numpy(action))(y), y0=torch.from_numpy(state),
                            t=torch.tensor([0.0, self.timestep]))[-1].detach().numpy()
        # self.state = self.agent.dynamics.state_to_mj_state(next_state, self.state)
        # TODO: rename to update
        self.state = self.agent.dynamics.next_state_to_mj_data_and_qpos(next_state=next_state,
                                                                        mj_state=self.state,
                                                                        data=self.data)



