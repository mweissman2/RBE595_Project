import AirSim.PythonClient.multirotor.setup_path as setup_path
import airsim
import gym
import pprint
from stable_baselines3 import DQN

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

state = client.getMultirotorState()
s = pprint.pformat(state)
# print("state: %s" % s)
# print(state.kinematics_estimated.position.x_val)

class AirSimEnv(gym.Env):
    def __init__(self):
        super(AirSimEnv, self).__init__()
        self.client = self.start_client()
        self.action_space = gym.spaces.Discrete(9)  # 9 discrete actions
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=float)  # Example state space (e.g., position)

        # Initialize environment variables
        self.start_position = (0, 0, 0)  # Starting position
        self.goal_position = (10, 10, 0)  # Goal position
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

    def reset_env(self):
        # Reset the environment to the starting state
        self.current_position = self.start_position
        return self.current_position

    def get_position(self):
        # Gets current position from robot state
        cur_state = client.getMultirotorState()
        x = cur_state.kinematics_estimated.position.x_val
        y = cur_state.kinematics_estimated.position.y_val
        z = cur_state.kinematics_estimated.position.z_val
        position = (x,y,z)
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

        return state, reward, done, {}

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
        return ((self.current_position[0] - self.goal_position[0]) ** 2 +
                (self.current_position[1] - self.goal_position[1]) ** 2 +
                (self.current_position[2] - self.goal_position[2]) ** 2) ** 0.5



env = AirSimEnv
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)