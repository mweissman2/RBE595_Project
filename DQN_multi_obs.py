import time

import AirSim.PythonClient.multirotor.setup_path as setup_path
import airsim
import gym
import gymnasium
import pprint
import numpy as np
import random
import transforms3d.quaternions as quaternions
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

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
        self.start_position = np.array([-55.0, -90.0, -4.0])
        self.goal_position = np.array([-33.0, -102.0, -4.0])
        self.current_position = self.start_position.copy()
        self.goal_threshold = 5.0
        self.initial_pose = self.client.simGetVehiclePose()
        self.initial_pose.position.x_val = self.start_position[0]
        self.initial_pose.position.y_val = self.start_position[1]
        self.initial_pose.position.z_val = self.start_position[2]
        self.vel_step = 2
        self.move_time = 0.3

        # Set point
        self.spawn_setpoint = True
        self.viz_offset = -15.0
        self.set_point_start_pos = self.start_position.copy()
        self.set_point_position = self.set_point_start_pos.copy()
        self.set_point_speed = self.vel_step
        self.magnitude = np.linalg.norm(np.array(self.goal_position) - np.array(self.set_point_start_pos))
        self.direction_vector = np.array([(self.goal_position[0] - self.set_point_start_pos[0])/self.magnitude,
                                          (self.goal_position[1] - self.set_point_start_pos[1])/self.magnitude,
                                          (self.goal_position[2] - self.set_point_start_pos[2])/self.magnitude])


        self.sim_initialization()

        self.action_space = gymnasium.spaces.Discrete(11)  # 10 discrete actions
        self.observation_space = gymnasium.spaces.Dict(
            {
            "position": gymnasium.spaces.Box(low=-150, high=150, shape=(3,), dtype=float),  # Position
            "image": gymnasium.spaces.Box(low=0, high=255, shape=(144,256,1), dtype=np.uint8),
            }
        )
        # self.observation_space = gymnasium.spaces.Box(low=-20, high=20, shape=(3,), dtype=float)
        # self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(144,256,1), dtype=np.uint8)

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
        # Stretch Cylinders
        for i in range(2, 9):
            name = "Cylinder" + str(i)
            cyl_scale = self.client.simGetObjectScale(name)
            cyl_scale.z_val = 30.0
            self.client.simSetObjectScale(name, cyl_scale)

        cyl_scale = self.client.simGetObjectScale('Cylinder_2')
        cyl_scale.z_val = 30.0
        self.client.simSetObjectScale('Cylinder_2', cyl_scale)

        # Spawn Goal
        if "Goal" not in self.client.simListSceneObjects():
            # self.client.simDestroyObject("Goal")
            spawn_pose = airsim.Pose()
            spawn_pose.position.x_val = self.goal_position[0]
            spawn_pose.position.y_val = self.goal_position[1]
            spawn_pose.position.z_val = self.goal_position[2]
            spawn_scale = airsim.Vector3r(0.3, 0.3, 0.3)
            self.client.simSpawnObject('Goal', 'Sphere', spawn_pose, spawn_scale, False)
        else:
            self.reset_goal()

        # Spawn setpoint
        if self.spawn_setpoint:
            if "Setpoint" not in self.client.simListSceneObjects():
                spawn_pose = airsim.Pose()
                spawn_pose.position.x_val = self.set_point_position[0]
                spawn_pose.position.y_val = self.set_point_position[1]
                spawn_pose.position.z_val = self.set_point_position[2] + self.viz_offset
                print("spawning setpoint")
                spawn_scale = airsim.Vector3r(0.5, 0.5, 0.5)
                self.client.simSpawnObject('Setpoint', 'Sphere', self.initial_pose, spawn_scale, False)
            else:
                self.reset_setpoint()
    def reset_setpoint(self):
        self.set_point_position = self.set_point_start_pos.copy()  # Reset set point pos

        if self.spawn_setpoint:
            temp_pose = airsim.Pose()
            temp_pose.position.x_val = self.set_point_position[0]
            temp_pose.position.y_val = self.set_point_position[1]
            temp_pose.position.z_val = self.set_point_position[2] + self.viz_offset
            self.client.simSetObjectPose('Setpoint', temp_pose, teleport=True)

    def reset_goal(self):
        # rand_num = 0
        # self.goal_position = np.array([-45.0, -57.0 + rand_num, -10.0])  # Reset set point pos

        temp_pose = airsim.Pose()
        temp_pose.position.x_val = self.goal_position[0]
        temp_pose.position.y_val = self.goal_position[1]
        temp_pose.position.z_val = self.goal_position[2]
        self.client.simSetObjectPose('Goal', temp_pose, teleport=True)

    def get_rotation_mat(self):
        quad_state = self.client.getMultirotorState()
        orientation_quat = quad_state.kinematics_estimated.orientation
        rotation_matrix = quaternions.quat2mat([orientation_quat.w_val, orientation_quat.x_val,
                                         orientation_quat.y_val, orientation_quat.z_val])
        return rotation_matrix


    def _get_obs(self):
        relative_pos = self.set_point_position - self.current_position
        R = self.get_rotation_mat()
        relative_pos_transformed = np.dot(R, relative_pos)
        depth = self.get_depth_img()
        return {"position": relative_pos_transformed, "image": depth, }

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # Reset the environment to the starting state
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.simSetVehiclePose(self.initial_pose, True)

        self.current_position = self.start_position.copy()

        self.reset_goal()
        self.reset_setpoint()

        observation = self._get_obs()  # Set initial observation
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
        old_distance = self.calculate_distance_to_setpoint()

        match action:
            case 0:
                self.client.moveByVelocityBodyFrameAsync(self.vel_step, -self.vel_step, -self.vel_step, self.move_time).join()
            case 1:
                self.client.moveByVelocityBodyFrameAsync(self.vel_step, 0, -self.vel_step, self.move_time).join()
            case 2:
                self.client.moveByVelocityBodyFrameAsync(self.vel_step, self.vel_step, -self.vel_step, self.move_time).join()
            case 3:
                self.client.moveByVelocityBodyFrameAsync(self.vel_step, -self.vel_step, 0, self.move_time).join()
            case 4:
                self.client.moveByVelocityBodyFrameAsync(self.vel_step, 0, 0, self.move_time).join()
            case 5:
                self.client.moveByVelocityBodyFrameAsync(self.vel_step, self.vel_step, 0, self.move_time).join()
            case 6:
                self.client.moveByVelocityBodyFrameAsync(self.vel_step, -self.vel_step, self.vel_step-1, self.move_time).join()
            case 7:
                self.client.moveByVelocityBodyFrameAsync(self.vel_step, 0, self.vel_step-1, self.move_time).join()
            case 8:
                self.client.moveByVelocityBodyFrameAsync(self.vel_step, self.vel_step, self.vel_step-1, self.move_time).join()
            # case 9:
            #     self.client.moveByVelocityBodyFrameAsync(0, 0, 0, self.move_time).join()
            case 9:
                self.client.moveByRollPitchYawZAsync(0, 0, -np.pi / 3, 0, 0.5)
            case 10:
                self.client.moveByRollPitchYawZAsync(0, 0, np.pi / 3, 0, 0.5)

        # Move setpoint
        self.move_setpoint()

        # Update current position
        self.current_position = self.get_position()

        # Find new distance to goal after move
        new_distance = self.calculate_distance_to_setpoint()

        # Get state (observations)
        obs = self._get_obs()

        # Initialize done flag
        done = False
        truncate = False

        # Check collision
        collision_check = self.client.simGetCollisionInfo().has_collided
        if collision_check:
            done = True
            print("COLLISION! Resetting...")

        # Calculate reward
        reward = self.get_reward(old_distance, new_distance, collision_check)
        # print(f'Observation: {obs}')

        # Check if episode is done (close to goal)
        distance_to_goal = self.calculate_distance_to_goal()
        goal_reached = (distance_to_goal < self.goal_threshold)
        if goal_reached:
            done = True
            reward = 100
            print("GOAL REACHED!")

        # if self.current_position[0] > self.goal_position[0]:
        #     truncate = True
        #     reward = -20

        info = self._get_info()  # dictionary for additional information

        print(f'Distance to setpoint: {new_distance}, Setpoint Position: {self.set_point_position}, Reward: {reward}')

        return obs, reward, done, truncate, info

    def get_reward(self, old_d, new_d, collision_check):
        del_d = new_d - old_d
        d_t = new_d

        # if collision_check:
        #     reward = -1
        # elif self.del_d_u < del_d:
        #     reward = self.R_l/d_t
        # elif del_d >= self.del_d_l and del_d <= self.del_d_u:
        #     reward = self.R_l + (self.R_u - self.R_l) * ((self.del_d_u - del_d) / (self.del_d_u - self.del_d_l))
        # elif del_d < self.del_d_l:
        #     reward = self.R_u/d_t
        # else:
        #     reward = self.R_dp


        # if collision_check:
        #     reward = -50
        #     chosen = 1
        # elif new_d < 2:
        #     reward = 0.5
        #     chosen = 2
        # else:
        #     reward = (1 / new_d**2)
        #     chosen = -1
        #
        # return reward


        if collision_check:
            reward = -50
        else:
            x_distance = abs(self.current_position[0] - self.set_point_position[0])
            y_distance = abs(self.current_position[1] - self.set_point_position[1])
            z_distance = abs(self.current_position[2] - self.set_point_position[2])
            reward = -((0.2 * x_distance) + (0.4 * y_distance) + (0.4 * z_distance))
        return reward

    def calculate_distance_to_goal(self):
        # Calculate distance from current position to goal
        return np.linalg.norm(np.array(self.current_position) - np.array(self.goal_position))

    def calculate_distance_to_setpoint(self):
        # Calculate distance from current position to set point
        return np.linalg.norm(np.array(self.current_position) - np.array(self.set_point_position))

    def move_setpoint(self):
        distance_to_goal = np.linalg.norm(np.array(self.set_point_position) - np.array(self.goal_position))
        # print(f'Distance from point to goal: {distance_to_goal}')
        if distance_to_goal >= 2:
            # self.set_point_position += self.direction_vector * (self.set_point_speed * self.move_time)
            self.set_point_position += self.direction_vector * (self.set_point_speed * 0.475)
        else:
            self.set_point_position = self.goal_position.copy()

        if self.spawn_setpoint:
            temp_pose = airsim.Pose()
            temp_pose.position.x_val = self.set_point_position[0].copy()
            temp_pose.position.y_val = self.set_point_position[1].copy()
            temp_pose.position.z_val = self.set_point_position[2].copy() + self.viz_offset
            self.client.simSetObjectPose('Setpoint', temp_pose)

    def get_depth_img(self):
        responses = self.client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthPlanar, pixels_as_float=True)])
        response = responses[0]
        img1d = np.array(response.image_data_float, dtype=np.float32)
        img1d = img1d * 3.5 + 30
        img1d[img1d > 255] = 255
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width,1))
        depth = np.array(img2d, dtype=np.uint8)
        return depth

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
    env = VecTransposeImage(env)

    # model = DQN("CnnPolicy", env, verbose=1, exploration_fraction=0.1)
    model = DQN(
        "MultiInputPolicy",
        env,
        # learning_rate=0.00025,
        learning_rate=0.1,
        verbose=1,
        batch_size=32,
        train_freq=4,
        target_update_interval=10000,
        learning_starts=10000,
        buffer_size=500000,
        max_grad_norm=10,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        # tensorboard_log="./tb_logs/",
    )
    print("Starting training...")
    model.learn(total_timesteps=10000, log_interval=5)
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