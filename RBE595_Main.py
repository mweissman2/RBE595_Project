# DISCLAIMER: Currently this code does not work unless it is in the directory 'AirSim\PythonClient\multirotor', we can fix that later. For now it is in the outermost location, which I think makes sense as the final location

import AirSim.PythonClient.multirotor.setup_path as setup_path
import airsim

import numpy as np
import os
import tempfile
import pprint
import cv2

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

state = client.getMultirotorState()
s = pprint.pformat(state)
print("state: %s" % s)

airsim.wait_key('Press any key to takeoff')
print("Taking off...")
client.armDisarm(True)
client.takeoffAsync().join()

state = client.getMultirotorState()
print("state: %s" % pprint.pformat(state))

# ----- Movement Testing -----
# https://microsoft.github.io/AirSim/api_docs/html/
# Move w.r.t. world: moveByVelocityAsync
# Move w.r.t. vehicle: moveByVelocityBodyFrameAsync
airsim.wait_key('Press any key to elevate')
client.moveByVelocityBodyFrameAsync(0, 0, -5, 5).join()

# Motion Primitives:
# - Forward, hold Alt ------ s
# - 45 deg left, hold Alt -- a
# - 45 deg right, hold Alt - d
# - Forward and Up --------- w
# - 45 deg left and up ----- q
# - 45 deg right and up ---- e
# - forward and down ------- x
# - 45 deg left and down --- z
# - 45 deg right and down -- c

# For now, we will define 'forward' as positive X
# Number of times we want to press the key before stopping
num_keys = 5
# Time in s to move
len_move = 2

for i in range(num_keys):
    key = airsim.wait_key('Press any key to move vehicle. See the code for the movement guide')
    ASKey = key.decode('ASCII')
    match ASKey:
        case 'q':
            client.moveByVelocityBodyFrameAsync(3, -3, -3, len_move).join()
        case 'w':
            client.moveByVelocityBodyFrameAsync(3, 0, -3, len_move).join()
        case 'e':
            client.moveByVelocityBodyFrameAsync(3, 3, -3, len_move).join()
        case 'a':
            client.moveByVelocityBodyFrameAsync(3, -3, 0, len_move).join()
        case 's':
            client.moveByVelocityBodyFrameAsync(3, 0, 0, len_move).join()
        case 'd':
            client.moveByVelocityBodyFrameAsync(3, 3, 0, len_move).join()
        case 'z':
            client.moveByVelocityBodyFrameAsync(3, -3, 2, len_move).join()
        case 'x':
            client.moveByVelocityBodyFrameAsync(3, 0, 2, len_move).join()
        case 'c':
            client.moveByVelocityBodyFrameAsync(3, 3, 2, len_move).join()


# ----- RESET ----- 
airsim.wait_key('Press any key to reset to original state')
client.reset()
client.armDisarm(False)
# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)