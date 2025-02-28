import AirSim.PythonClient.multirotor.setup_path as setup_path
import airsim
import gymnasium
import transforms3d.quaternions as quaternions
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

class AirSimEnv(gymnasium.Env):
    def __init__(self):
        super(AirSimEnv, self).__init__()
        self.client = self.start_client()
        self.action_space = gymnasium.spaces.Discrete(9)  # 9 discrete actions
        self.observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(3,), dtype=float)  # Example state space (e.g., position)

        # Initialize environment variables
        self.start_position = np.array([-5.0, 0.0, -1.0])
        self.goal_position = np.array([11.0, 0.0, -1.0])
        self.current_position = self.start_position
        self.goal_threshold = 1.0
        self.initial_pose = self.client.simGetVehiclePose()
        self.sim_initialization()

        # Moving Setpoint Variables
        self.magnitude = np.linalg.norm(np.array(self.goal_position) - np.array(self.start_position))
        self.direction_vector = np.array([(self.goal_position[0] - self.start_position[0])/self.magnitude, 
                                          (self.goal_position[1] - self.start_position[1])/self.magnitude, 
                                          (self.goal_position[2] - self.start_position[2])/self.magnitude])
        self.setpoint = np.array([self.start_position[0], self.start_position[1], self.start_position[2]])
        self.setpoint_threshold = 2.0
        
        # Action Variables
        self.vel_max = 2
        self.move_time = 0.2

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
        Spawns goal
        """
        self.initial_pose.position.x_val = self.start_position[0]
        self.initial_pose.position.y_val = self.start_position[1]
        self.initial_pose.position.z_val = self.start_position[2]

        # Spawn Goal
        spawn_pose = airsim.Pose()
        spawn_pose.position.x_val = self.goal_position[0]
        spawn_pose.position.y_val = self.goal_position[1]
        spawn_pose.position.z_val = self.goal_position[2]
        spawn_scale = airsim.Vector3r(self.goal_threshold, self.goal_threshold, self.goal_threshold)
        self.client.simSpawnObject('Goal', 'Sphere', spawn_pose, spawn_scale, False)

    def get_position(self):
        """
        Calculates the current state position of the multirotor
        Return: Position np array (3,1)
        """
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

    def calculate_distance(self, point):
        # Calculate distance from current position to goal
        return np.linalg.norm(np.array(self.current_position) - np.array(point))

    def calculate_distance_to_goal(self):
        # Calculate distance from current position to goal
        return np.linalg.norm(np.array(self.current_position) - np.array(self.goal_position))
    
    def _get_obs(self):
        relative_pos = self.setpoint - self.current_position
        R = self.get_rotation_mat()
        relative_pos_transformed = np.dot(R, relative_pos)
        return relative_pos_transformed
    
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
        self.current_position = np.array([self.start_position[0], self.start_position[1], self.start_position[2]])
        self.setpoint = np.array([self.start_position[0], self.start_position[1], self.start_position[2]])
        observation = self.current_position  # Set initial observation
        info = self._get_info()  # dictionary for additional information
        return observation, info

    def step(self, action):       
        """
        Main function for RL algorithm. Executes action, collects observation, calculates reward, and termination flags
        Return: observation dict, reward, termination flag, truncation flag, info dict
        """
        # Get current distance to goal before move
        old_distance = self.calculate_distance_to_goal()

        match action:
            case 0:
                self.client.moveByVelocityBodyFrameAsync(self.vel_max, -self.vel_max, -self.vel_max, self.move_time).join()
            case 1:
                self.client.moveByVelocityBodyFrameAsync(self.vel_max, 0, -self.vel_max, self.move_time).join()
            case 2:
                self.client.moveByVelocityBodyFrameAsync(self.vel_max, self.vel_max, -self.vel_max, self.move_time).join()
            case 3:
                self.client.moveByVelocityBodyFrameAsync(self.vel_max, -self.vel_max, 0, self.move_time).join()
            case 4:
                self.client.moveByVelocityBodyFrameAsync(self.vel_max, 0, 0, self.move_time).join()
            case 5:
                self.client.moveByVelocityBodyFrameAsync(self.vel_max, self.vel_max, 0, self.move_time).join()
            case 6:
                self.client.moveByVelocityBodyFrameAsync(self.vel_max, -self.vel_max, self.vel_max, self.move_time).join()
            case 7:
                self.client.moveByVelocityBodyFrameAsync(self.vel_max, 0, self.vel_max, self.move_time).join()
            case 8:
                self.client.moveByVelocityBodyFrameAsync(self.vel_max, self.vel_max, self.vel_max, self.move_time).join()
            case _:
                print('Error')

        # Update current position
        self.current_position = self.get_position()

        # Find new distance to goal after move
        goal_distance = self.calculate_distance_to_goal()
        new_distance = self.calculate_distance(self.setpoint)

        # Get state (observations)
        state = self._get_obs()

        # Check for collisions
        collision_check = self.client.simGetCollisionInfo().has_collided
        if collision_check:
            print("COLLISION! Resetting...")

        # Calculate reward
        reward, chosen = self.get_reward(old_distance, new_distance, goal_distance, collision_check)
        print(f'Distance to go: {new_distance}, Reward: {reward}, Setpoint: {self.setpoint}, Position: {self.current_position}')

        # Calculate the Moving Setpoint
        if np.linalg.norm(np.array(self.setpoint) - np.array(self.goal_position)) > 0.05:
            self.setpoint += self.direction_vector * (self.vel_max * self.move_time)

        # Check if episode is done (close to goal)
        done = (goal_distance < self.goal_threshold)
        if done:
            print("GOAL REACHED!")
            reward = 20

        info = self._get_info()  # dictionary for additional information

        if done or collision_check:
            term = True
        else:
            term = False

        return state, reward, term, False, info

    def get_reward(self, old_d, new_d, goal_d, collision_check):        
        """
        Calculates the reward using the current position and setpoint position
        Return: reward value (float)
        """
        if collision_check:
            reward = -10
            chosen = 1
        elif new_d < 2:
            reward = 0.5
            chosen = 2
        else:
            reward = .1 - (1 / new_d)
            chosen = -1  
        return reward, chosen

def main():
    # Register the environment
    gymnasium.register(
        id='AirSimEnv-v0',
        entry_point=lambda: AirSimEnv(),
        max_episode_steps=65
    )

    temp_env = gymnasium.make('AirSimEnv-v0')
    env = DummyVecEnv([lambda: temp_env])

    model = DQN("MlpPolicy", env, verbose=1)
    airsim.wait_key('Press any key to start training')
    model.learn(total_timesteps=1500, log_interval=4)
    print("Training complete!")

    # Play after training
    airsim.wait_key('Press any key to enter prediction phase')
    model.save('DQN-No_obstacle-v1')
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated = env.step(action)
        if terminated:
            print('RESET')
            obs = env.reset()


if __name__ == '__main__':
    main()