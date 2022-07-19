"""
Visualize the view from the PTZ camera of the car
Change the below fields to connect to the desired vehicle
"""

global_port = 41451
vehicle_name = "Car0"
camera_name = "0"


import airsim
import cv2
import numpy as np
import os
import time
import pprint
from airsim.types import Pose
from airsim import Vector3r, Quaternionr, Pose
import _thread
import cv2

from PIL import Image


def draw_bounding_box(client, img_rgb):
    cars = client.simGetDetections(camera_name="0", image_type=0)
    if cars:
        car = cars[0]
        x1 = int(car.box2D.min.x_val)
        y1 = int(car.box2D.min.y_val)
        x2 = int(car.box2D.max.x_val)
        y2 = int(car.box2D.max.y_val)
        cv2.rectangle(img_rgb,(x1,y1),(x2,y2),(255,0,0),2)

    return img_rgb

def convert_rgb_to_gray(observation):
    r, g, b = observation[:, :, 0], observation[:, :, 1], observation[:, :, 2]
    observation = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return observation


def capture_camera(client, vehicle_name, camera_name):

    while(True):

        responses = client.simGetImages([airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)],vehicle_name=vehicle_name)  #scene vision image in uncompressed RGB array
        image_data = responses[0]
        image = Image.frombytes('RGB', (image_data.width, image_data.height), image_data.image_data_uint8, 'raw', 'RGB', 0, 1)
        img_rgb = np.array(image)

        cv2.imshow(str(global_port), img_rgb)
        cv2.waitKey(100)


# connect to the AirSim simulator
client1 = airsim.CarClient(port = global_port)
client1.confirmConnection()

capture_camera(client1,vehicle_name, camera_name)
