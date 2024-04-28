from DQN import airsim
import time
import numpy as np
import cv2


client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

# print(client.simListSceneObjects())

pose = client.simGetVehiclePose()
pose.position.x_val = -25.0
client.simSetVehiclePose(pose, True)

responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthPlanar, pixels_as_float=True)])
response = responses[0]
img1d = np.array(response.image_data_float, dtype=np.float32)
img1d = img1d * 3.5 + 30
img1d[img1d > 255] = 255
img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
depth = np.array(img2d, dtype=np.uint8)

cv2.imshow('Airsim',depth)
cv2.waitKey(0)