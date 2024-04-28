import time

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

        # Initialize environment variables
        self.start_position = np.array([0.0, 0.0, 0.0])
        self.goal_position = np.array([7.0, 0.0, -5.0])
        self.current_position = self.start_position
        self.goal_threshold = 2.0
        self.initial_pose = self.client.simGetVehiclePose()
        self.sim_initialization()
        self.move_time = 0.5

        self.action_space = gymnasium.spaces.Discrete(10)  # 10 discrete actions
        # self.observation_space = gymnasium.spaces.Dict(
        #     {
        #     "position": gymnasium.spaces.Box(low=0, high=2, shape=(3,), dtype=float)  # Position
        #     "image": spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8)
        #     }
        # )
        self.observation_space = gymnasium.spaces.Box(low=-20, high=20, shape=(3,), dtype=float)

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

    def sim_initialization(self):
        self.initial_pose.position.x_val = -25.0
        self.initial_pose.position.z_val = -1.0

        # Spawn Goal
        spawn_pose = airsim.Pose()
        spawn_pose.position.x_val = self.goal_position[0]
        spawn_pose.position.z_val = self.goal_position[2]
        spawn_scale = airsim.Vector3r(2.0, 2.0, 2.0)
        self.client.simSpawnObject('Goal', 'Sphere', spawn_pose, spawn_scale, False)

    def _get_obs(self):
        return self.current_position

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # Reset the environment to the starting state
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.simSetVehiclePose(self.initial_pose, True)

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
            case 9:
                self.client.moveByVelocityBodyFrameAsync(0, 0, 0, self.move_time).join()

        # Update current position
        self.current_position = self.get_position()

        # Find new distance to goal after move
        new_distance = self.calculate_distance_to_goal()

        # Get state (observations)
        obs = self._get_obs()

        done = False

        collision_check = self.client.simGetCollisionInfo().has_collided
        if collision_check:
            done = True
            print("COLLISION! Resetting...")

        # Calculate reward
        reward = self.get_reward(old_distance, new_distance, collision_check)
        print(f'Distance to goal: {new_distance}, Reward: {reward}')

        # Check if episode is done (close to goal)
        goal_reached = (new_distance < self.goal_threshold)
        if goal_reached:
            done = True
            print("GOAL REACHED!")

        info = self._get_info()  # dictionary for additional information

        return obs, reward, done, False, info

    def get_reward(self, old_d, new_d, collision_check):
        del_d = new_d - old_d
        # d_t = new_d
        # if self.del_d_u < del_d:
        #     return self.R_l/d_t
        print(f'New Distance: {new_d}, Old Distance: {old_d}')

        if collision_check:
            return -10
        if new_d < self.goal_threshold:
            return 100
        # else:
        #     return -0.01
        # else:
        #     return 1/new_d
        elif old_d <= new_d:
            return -1
        else:
            return 1

    def calculate_distance_to_goal(self):
        # Calculate distance from current position to goal
        return np.linalg.norm(np.array(self.current_position) - np.array(self.goal_position))

    def close_stream(self):
        # Close AirSim client connection
        self.client.reset()
        self.client.enableApiControl(False)



def main():
    # Register the environment
    gymnasium.register(
        id='AirSimEnv-v0',
        entry_point=lambda: AirSimEnv(),
        max_episode_steps=40,
    )
    # gym.envs.register(id='AirSimEnv-v0', entry_point=AirSimEnv)
    temp_env = gymnasium.make('AirSimEnv-v0')
    env = DummyVecEnv([lambda: temp_env])

    model = DQN("MlpPolicy", env, verbose=1, exploration_fraction=0.1)
    print("Starting training...")
    model.learn(total_timesteps=300, log_interval=4)
    print("Training complete!")

    model.save('DQN-No_obstacle-v1')

    # Play after training
    airsim.wait_key('Press any key to reset to original state')
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated = env.step(action)
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        if terminated:
            obs = env.reset()


if __name__ == '__main__':
    main()