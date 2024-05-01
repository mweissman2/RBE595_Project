import AirSim.PythonClient.multirotor.setup_path as setup_path
import airsim
import gymnasium
import numpy as np
import random
import transforms3d.quaternions as quaternions
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

class AirSimEnv(gymnasium.Env):
    def __init__(self):
        super(AirSimEnv, self).__init__()
        self.client = self.start_client()

        # Initialize environment variables
        self.start_position = np.array([-100.0, -57.0, -10.0])
        self.goal_position = np.array([-45.0, -57.0, -10.0])
        self.current_position = self.start_position.copy()
        self.goal_threshold = 5.0
        self.initial_pose = self.client.simGetVehiclePose()
        self.initial_pose.position.x_val = self.start_position[0]
        self.initial_pose.position.y_val = self.start_position[1]
        self.initial_pose.position.z_val = self.start_position[2]
        self.vel_step = 10.0
        self.move_time = 0.4

        # Set point
        self.spawn_setpoint = True
        self.viz_offset = -15.0
        self.set_point_start_pos = np.array([-100.0, -57.0, -10.0])
        self.set_point_position = self.set_point_start_pos.copy()
        self.set_point_speed = self.vel_step
        self.magnitude = np.linalg.norm(np.array(self.goal_position) - np.array(self.set_point_start_pos))
        self.direction_vector = np.array([(self.goal_position[0] - self.set_point_start_pos[0])/self.magnitude,
                                          (self.goal_position[1] - self.set_point_start_pos[1])/self.magnitude,
                                          (self.goal_position[2] - self.set_point_start_pos[2])/self.magnitude])

        # Setup simulation geometry components
        self.sim_initialization()

        # Define action and state space
        self.action_space = gymnasium.spaces.Discrete(9)  # 10 discrete actions
        self.observation_space = gymnasium.spaces.Dict(
            {
            "position": gymnasium.spaces.Box(low=-150, high=150, shape=(3,), dtype=float),  # Position
            "image": gymnasium.spaces.Box(low=0, high=255, shape=(144,256,1), dtype=np.uint8),
            }
        )

        # Set up rewards from paper
        self.R_l = 0.0
        self.R_u = 0.5
        self.R_dp = -0.5
        self.R_cp = -1
        self.del_d_l = -1
        self.del_d_u = 1

    @staticmethod
    def start_client():
        """
        Starts the Airsim Client
        """
        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)
        return client

    def sim_initialization(self):
        """
        Spawns goal, spawns setpoint
        """
        # Spawn Goal
        if "Goal" in self.client.simListSceneObjects():
            self.client.simDestroyObject("Goal")
        spawn_pose = airsim.Pose()
        spawn_pose.position.x_val = self.goal_position[0]
        spawn_pose.position.y_val = self.goal_position[1]
        spawn_pose.position.z_val = self.goal_position[2]
        spawn_scale = airsim.Vector3r(0.3, 0.3, 0.3)
        self.client.simSpawnObject('Goal', 'Sphere', spawn_pose, spawn_scale, False)

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
        """
        Resets setpoint to its start position (with visualization offset)
        """
        self.set_point_position = self.set_point_start_pos.copy()  # Reset set point pos

        if self.spawn_setpoint:
            temp_pose = airsim.Pose()
            temp_pose.position.x_val = self.set_point_position[0]
            temp_pose.position.y_val = self.set_point_position[1]
            temp_pose.position.z_val = self.set_point_position[2] + self.viz_offset
            self.client.simSetObjectPose('Setpoint', temp_pose, teleport=True)

    def reset_goal(self):
        """
        Resets goal to its start position (with random y offset)
        """
        rand_num = random.randint(-5,5)
        self.goal_position = np.array([-45.0, -57.0 + rand_num, -10.0])  # Reset set point pos

        temp_pose = airsim.Pose()
        temp_pose.position.x_val = self.goal_position[0]
        temp_pose.position.y_val = self.goal_position[1]
        temp_pose.position.z_val = self.goal_position[2]
        self.client.simSetObjectPose('Goal', temp_pose, teleport=True)

    def get_position(self):
        """
        Calculates the current state position of the multirotor
        Return: Position np array (3,1)
        """
        # Gets current position from robot state
        cur_state = self.client.getMultirotorState()
        x = cur_state.kinematics_estimated.position.x_val
        y = cur_state.kinematics_estimated.position.y_val
        z = cur_state.kinematics_estimated.position.z_val
        position = np.array([x,y,z])
        return position

    def get_rotation_mat(self):
        """
        Calculates rotation matrix for current multirotor state
        Return: Rotation Matrix
        """
        quad_state = self.client.getMultirotorState()
        orientation_quat = quad_state.kinematics_estimated.orientation
        rotation_matrix = quaternions.quat2mat([orientation_quat.w_val, orientation_quat.x_val,
                                         orientation_quat.y_val, orientation_quat.z_val])
        return rotation_matrix

    def calculate_distance_to_goal(self):
        """
        Calculates the distance from the multirotor to the goal
        Return: distance (float)
        """
        return np.linalg.norm(np.array(self.current_position) - np.array(self.goal_position))

    def calculate_distance_to_setpoint(self):
        """
        Calculates the distance from the multirotor to the setpoint
        Return: distance (float)
        """
        return np.linalg.norm(np.array(self.current_position) - np.array(self.set_point_position))

    def move_setpoint(self):
        """
        Increments the setpoint towards the goal (and teleports setpoint visualization)
        """
        distance_to_goal = np.linalg.norm(np.array(self.set_point_position) - np.array(self.goal_position))

        if distance_to_goal >= 4:
            self.set_point_position += self.direction_vector * (self.set_point_speed * 0.4)
        else:
            self.set_point_position = self.goal_position.copy()

        if self.spawn_setpoint:
            temp_pose = airsim.Pose()
            temp_pose.position.x_val = self.set_point_position[0].copy()
            temp_pose.position.y_val = self.set_point_position[1].copy()
            temp_pose.position.z_val = self.set_point_position[2].copy() + self.viz_offset
            self.client.simSetObjectPose('Setpoint', temp_pose)

    def get_depth_img(self):
        """
        Calculates the distance from the multirotor to the goal
        Return: depth image np array (144,256,1)
        """
        # Call API to get depth planar image
        responses = self.client.simGetImages(
            [airsim.ImageRequest(0, airsim.ImageType.DepthPlanar, pixels_as_float=True)])
        response = responses[0]

        # Convert to float32 and threshold depth values
        img1d = np.array(response.image_data_float, dtype=np.float32)
        img1d = img1d * 3.5 + 30
        img1d[img1d > 255] = 255

        # Reshape and convert to uint8
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 1))
        depth = np.array(img2d, dtype=np.uint8)
        return depth

    def _get_obs(self):
        """
        Collects observation dictionary: relative position difference and depth image
        Return: Observation Dictionary
        """
        relative_pos = self.set_point_position - self.current_position
        R = self.get_rotation_mat()
        relative_pos_transformed = np.dot(R, relative_pos)
        depth = self.get_depth_img()
        return {"position": relative_pos_transformed, "image": depth, }

    def _get_info(self):
        """
        Collects any additional state information
        """
        return {}

    def reset(self, seed=None, options=None):
        """
        Resets the environment after episode end
        Return: Observation dict and info dict
        """
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

    def step(self, action):
        """
        Main function for RL algorithm. Executes action, collects observation, calculates reward, and termination flags
        Return: observation dict, reward, termination flag, truncation flag, info dict
        """
        # Get current distance to goal before move
        old_distance = self.calculate_distance_to_setpoint()

        # Execute action
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

        # Check if episode is done (close to goal)
        distance_to_goal = self.calculate_distance_to_goal()
        goal_reached = (distance_to_goal < self.goal_threshold)
        if goal_reached:
            done = True
            reward = 100
            print("GOAL REACHED!")

        # Boundary check for truncation
        if self.current_position[0] > self.goal_position[0]:
            truncate = True
            reward = -20

        info = self._get_info()  # dictionary for additional information

        print(f'Distance to setpoint: {new_distance}, Setpoint Position: {self.set_point_position}, Reward: {reward}')

        return obs, reward, done, truncate, info

    def get_reward(self, old_d, new_d, collision_check):
        """
        Calculates the reward using the current positiona dnd setpoint position
        Return: reward value (float)
        """
        # Uncomment for paper reward system
        # del_d = new_d - old_d
        # d_t = new_d
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

        # Component based reward system
        if collision_check:
            reward = -50
        else:
            x_distance = abs(self.current_position[0] - self.set_point_position[0])
            y_distance = abs(self.current_position[1] - self.set_point_position[1])
            z_distance = abs(self.current_position[2] - self.set_point_position[2])
            reward = -((0.2 * x_distance) + (0.4 * y_distance) + (0.4 * z_distance))
        return reward

def main():
    # Register and set up the environment
    gymnasium.register(
        id='AirSimEnv-v0',
        entry_point=lambda: AirSimEnv(),
        max_episode_steps=30,
    )
    temp_env = gymnasium.make('AirSimEnv-v0')
    env = DummyVecEnv([lambda: temp_env])
    env = VecTransposeImage(env)

    # Create DQN model
    model = DQN(
        "MultiInputPolicy",
        env,
        # learning_rate=0.00025,
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
    model.learn(total_timesteps=1000, log_interval=5)
    print("Training complete!")

    # Save model
    model.save('DQN-No_obstacle-v1')

    # Play with optimal policy after training
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