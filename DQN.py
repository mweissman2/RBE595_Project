import AirSim.PythonClient.multirotor.setup_path as setup_path
import airsim
import gym
import gymnasium
import pprint
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# connect to the AirSim simulator
# client = airsim.MultirotorClient()
# client.confirmConnection()
# client.enableApiControl(True)
#
# state = client.getMultirotorState()
# s = pprint.pformat(state)
# print("state: %s" % s)
# print(state.kinematics_estimated.position.x_val)

class AirSimEnv(gymnasium.Env):
    def __init__(self):
        super(AirSimEnv, self).__init__()
        self.client = self.start_client()
        self.action_space = gymnasium.spaces.Discrete(9)  # 9 discrete actions
        self.observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(3,), dtype=float)  # Example state space (e.g., position)

        # Initialize environment variables
        self.start_position = np.array([0.0, 0.0, 0.0])
        self.goal_position = np.array([10.0, 10.0, 0.0])
        self.current_position = self.start_position

        self.move_time = 2

        # Set up rewards
        self.R_l = 0.0
        self.R_u = 0.5
        self.R_dp = -0.5
        self.R_cp = -1
        self.del_d_l = -1
        self.del_d_u = 1

    @staticmethod
    def start_client():
        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)
        return client

    def _get_obs(self):
        return {}

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # Reset the environment to the starting state
        self.current_position = self.start_position
        observation = self.current_position  # Set initial observation
        info = self._get_info()  # dictionary for additional information
        return observation, info

    def get_position(self):
        # Gets current position from robot state
        cur_state = self.client.getMultirotorState()
        x = cur_state.kinematics_estimated.position.x_val
        y = cur_state.kinematics_estimated.position.y_val
        z = cur_state.kinematics_estimated.position.z_val
        position = np.array([x,y,z])
        return position

    def step(self, action):
        # Perform action and return next state, reward, done, info

        # Get current distance to goal before move
        old_distance = self.calculate_distance_to_goal()

        match action:
            case 0:
                self.client.moveByVelocityBodyFrameAsync(3, -3, -3, self.move_time).join()
            case 1:
                self.client.moveByVelocityBodyFrameAsync(3, 0, -3, self.move_time).join()
            case 2:
                self.client.moveByVelocityBodyFrameAsync(3, 3, -3, self.move_time).join()
            case 3:
                self.client.moveByVelocityBodyFrameAsync(3, -3, 0, self.move_time).join()
            case 4:
                self.client.moveByVelocityBodyFrameAsync(3, 0, 0, self.move_time).join()
            case 5:
                self.client.moveByVelocityBodyFrameAsync(3, 3, 0, self.move_time).join()
            case 6:
                self.client.moveByVelocityBodyFrameAsync(3, -3, 2, self.move_time).join()
            case 7:
                self.client.moveByVelocityBodyFrameAsync(3, 0, 2, self.move_time).join()
            case 8:
                self.client.moveByVelocityBodyFrameAsync(3, 3, 2, self.move_time).join()

        # Update current position
        self.current_position = self.get_position()

        # Find new distance to goal after move
        new_distance = self.calculate_distance_to_goal()

        # Get state (observations)
        state = self.current_position

        # Calculate reward
        reward = self.get_reward(old_distance, new_distance)

        # Check if episode is done (close to goal)
        done = (new_distance < 1.0)

        info = self._get_info()  # dictionary for additional information

        return state, reward, done, False, info

    def get_reward(self, old_d, new_d):
        # del_d = new_d - old_d
        # d_t = new_d
        # if self.del_d_u < del_d:
        #     return self.R_l/d_t
        if new_d < 1.0:
            return 5
        else:
            return -0.01

    def calculate_distance_to_goal(self):
        # Calculate distance from current position to goal
        return np.linalg.norm(np.array(self.current_position) - np.array(self.goal_position))

    def close_stream(self):
        # Close AirSim client connection
        self.client.reset()
        self.client.enableApiControl(False)




# Register the environment
gymnasium.register(
    id='AirSimEnv-v0',
    entry_point=lambda: AirSimEnv()
)
# gym.envs.register(id='AirSimEnv-v0', entry_point=AirSimEnv)
temp_env = gymnasium.make('AirSimEnv-v0')
env = DummyVecEnv([lambda: temp_env])

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)