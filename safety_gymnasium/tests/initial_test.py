import safety_gymnasium
import mujoco
from scipy.spatial.transform import Rotation as R
import numpy as np

env = safety_gymnasium.make("SafetyPointunicycleGoal5-v0", render_mode="human", max_episode_steps=2000)
# env = safety_gymnasium.vector.make("SafetyCarGoal1-v0", render_mode="human", num_envs=8)

observation, info = env.reset(seed=0)

# vel = 1
# omega = 0.2
# dt = 0.01
# r = R.from_matrix(env.task.data.body('agent').xmat.reshape(3, 3))
# angle0 = r.as_euler('xyz')


for _ in range(1000):
    # action = env.action_space.sample()  # this is where you would insert your policy
    action = np.array([5.0, 4.0])
    observation, reward, cost, terminated, truncated, info = env.step(action)
   # r = R.from_matrix(env.task.data.body('agent').xmat.reshape(3, 3))
   # euler_angles = r.as_euler('xyz')
   # print(euler_angles)
   # pos = env.task.data.qpos


   # pos[0] += np.cos(angle0[2]) * vel * dt * np.cos(euler_angles[2]+ omega * dt / 2)
   # pos[1] += np.sin(-angle0[2]) * vel * dt * np.sin(euler_angles[2] + omega * dt / 2)
   # pos[2] += omega * dt
   # pos[0] += 0.05
   # pos[2] += 0.1
   # model = env.task.model
   # env.task.data.qpos[:] = pos
   # env.task.data.qvel[:] = vel
   # mujoco.mj_forward(model, env.task.data)
   # print(env.task.agent.vel)
   # env.render()


   # action = env.action_space.sample()  # this is where you would insert your policy
   # observation, reward, cost, terminated, truncated, info = env.step(action)

   # if terminated or truncated:
   #    observation, info = env.reset()

env.close()
