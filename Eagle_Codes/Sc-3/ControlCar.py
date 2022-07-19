"""
Control Car on random trajectories
global_port: Where the virtual world is running: Set in settings.json file
car_name: Name of the car to control: Set in settings.json file
"""

global_port = 41452
car_name = "Car1"

print('Running on port:', global_port)

import airsim
import cv2
import numpy as np
import os
import time
import pprint
from airsim.types import Pose
from airsim import Vector3r, Quaternionr, Pose
import random
import math


car_pos = None
car_state = None
count_struck = 0
count_stopped =0

client = airsim.CarClient(port = global_port)
client.confirmConnection()
client.enableApiControl(True,car_name)
client.reset()

y_min = -36
x_min = -8.0
y_max = 35.0
x_max = 58

fix_time_trajectory = 200

random.seed(0)


def initializeCar(client):
    #apply break to car-0
    car_controls_init = airsim.CarControls()
    car_controls_init.handbrake = True;
    car_controls_init.brake = 1.0
    client.setCarControls(car_controls_init, 'Car0')



def clamp(num, min_value, max_value):
    return max(min(num, max_value), min_value)


car_controls = airsim.CarControls()
def applyaction(move_forward, steering):
    steering = clamp(steering, -1.0, 1.0)

    if move_forward==1.0:
        car_controls.is_manual_gear = False;
        car_controls.throttle= 1.0
        car_controls.steering = steering
        client.setCarControls(car_controls, car_name)

    else:
        car_controls.is_manual_gear = True;
        car_controls.manual_gear = -1
        car_controls.throttle= 1.0
        client.setCarControls(car_controls, car_name)

def car_near_boundary(x, y, z):
    global x_min, x_max, y_min, y_max
    tolerance = 5

    if x<(x_min+tolerance) or x>(x_max-tolerance):
        return True

    if y<(y_min+tolerance) or y>(y_max-tolerance):
        return True

    return False

def is_struck():
    global count_struck, car_pos, car_state

    x = car_pos.position.x_val
    y = car_pos.position.y_val
    z = car_pos.position.z_val

    if abs(car_state.speed)<0.1 and car_near_boundary(x,y,z):
        count_struck+=1

    else:
        count_struck = 0

    if count_struck>8 and car_near_boundary(x,y,z):
        count_struck = 0
        return True

    return False

def is_stopped():
    global count_stopped, car_pos, car_state

    x = car_pos.position.x_val
    y = car_pos.position.y_val
    z = car_pos.position.z_val

    if abs(car_state.speed)<0.01:
        #print('car count_stopped:', count_stopped)
        count_stopped+=1

    else:
        count_stopped=0

    if count_stopped>40:
        count_stopped = 0

        print('x,y:',x,y)
        return True

    return False


def is_outside_world():
    global car_pos
    global x_min, x_max, y_min, y_max
    tolerance = 4
    x_min2=x_min - tolerance
    x_max2=x_max + tolerance

    y_min2=y_min - tolerance
    y_max2=y_max + tolerance


    x = car_pos.position.x_val
    y = car_pos.position.y_val
    z = car_pos.position.z_val

    #write here
    if x<(x_min2) or x>(x_max2):
        return True

    if y<(y_min2) or y>(y_max2):
        return True

    if abs(z) > 4:
        return True

    return False

def getCarState(car_name):
    global car_state
    car_state = client.getCarState(car_name)

def getCarPos(car_name):
    global car_pos
    car_pos = client.simGetObjectPose(car_name)


st_time = time.time()
steering = random.uniform(-0.2, 0.2)
move_forward = 1.0

while True:
    getCarState(car_name)
    getCarPos(car_name)
    applyaction(move_forward, steering)

    if (time.time() - st_time)>fix_time_trajectory:
        steering = random.uniform(-0.2, 0.2)
        move_forward  = 1.0
        st_time = time.time()
        #print('fix_time_trajectory True: move_forward, steering:', move_forward, steering)

    if is_struck():
        steering = random.uniform(-1.0, 1.0)
        move_forward = -1.0*move_forward
        #print('is_struck True: move_forward, steering:', move_forward, steering)

    #is car Outside the world
    if is_outside_world():
        print('*'*100)
        print('Outside World')
        client.reset()

    if is_stopped():
        print('*'*10)
        print('is_stopped')
        #client.reset()

    #print('Time passed:', time.time() - st_time)
    time.sleep(0.1)
