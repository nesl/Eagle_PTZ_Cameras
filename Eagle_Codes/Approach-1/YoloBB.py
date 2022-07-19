"""
Yolo Bounding-Box + Tuned-Kalman + Tuned-Controller
"""


no_of_episodes = 100
global_port = 41452

enable_debug = False
image_size_to_use = 240

save_name = 'evaluate.pickle'
print(save_name)


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


#load the yolo5s model
import torch
path = 'yolo5s_tuned.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=path)  # local model
print('loaded torch local model:', path)
model.cuda()
#ids from yolo5s to use
ids_obj = [2, 5, 7]


# see the proper environment is used
import sys
print(sys.executable)


import gym
import sys
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import numpy as np
from gym import spaces
from PIL import Image

import airsim
import cv2
import numpy as np
import os
import time
import pprint
from airsim.types import Pose
from airsim import Vector3r, Quaternionr, Pose
import random
import _thread


import matplotlib.pyplot as plt
import math

import random
random.seed(0)


### Gym Env ###
class PTZEnv(gym.Env):
    #camera starting pose
    camera_height = -8.0
    camera_x = 7.0
    camera_y = -50.0

    #camera initial fov
    initial_fov = 60
    initial_angleh = 0
    initial_anglev = -10.0

    #state space size
    screen_height = image_size_to_use
    screen_width = image_size_to_use
    captured_image = image_size_to_use
    seg_img = image_size_to_use

    #camera is attached t0 o Car0
    camera_name = "0"
    vehicle_name="Car0"
    camera_pos = None

    #scene camera
    camera_name2 = "1"

    #store the current camera values
    angleh = initial_angleh
    anglev = initial_anglev
    fov = initial_fov
    delta = 2.0

    curr_captured_cv_image = None
    curr_detections_engine = None

    def __init__(self, start_airsim=True, env_id = 0):

        # actions -> PTZ
        # actions -> PTZ, 3 for pan, 3 for tilt, 3 for zoom
        self.action_space = spaces.MultiDiscrete([ 3, 3, 3]) #spaces.Discrete(11)

        self.observation_space = spaces.Box(low=0, high=255,shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8)
        self.steps =0

        #records if camera was modified
        self.modified_cam = False

        #used to store one past state to be used when image collection fails
        self.past_image_state = None

        #if car is outside of boundary
        self.outside_car = False

        self.port = global_port+env_id

        print('Running on port:', self.port)

        if start_airsim:
            self.client = airsim.CarClient(port = self.port)
            self.initializeEnv()
            self.client_det = airsim.CarClient(port = self.port)
            self.client_det.simAddDetectionFilterMeshName(camera_name=self.camera_name, image_type=0, mesh_name="Car1",vehicle_name = self.vehicle_name)

        self.reward_x = 0
        self.reward_y = 0
        self.reward_object_size = 0
        self.curr_seg_image = None

        if enable_debug:
            #initialize the upper camera
            self.initialize_upper_camera()

    def initialize_upper_camera(self):
        client = self.client
        vehicle_name = self.vehicle_name
        camera_name = self.camera_name2

        camera_height = -70
        anglev = -58
        angleh = 0
        initial_fov = 60

        camera_pos = Pose()
        camera_pos.position.x_val= self.camera_x
        camera_pos.position.y_val= self.camera_y
        camera_pos.position.z_val= camera_height

        tilt = anglev/57.2958
        pan = angleh/57.2958

        camera_pos.orientation = airsim.to_quaternion(tilt , 0,  pan)
        success = client.simSetCameraPose(camera_name, camera_pos, vehicle_name)
        client.simSetCameraFov(camera_name, initial_fov, vehicle_name)


    def initializeEnv(self):

        client = self.client

        client.reset()

        #Iniliaze the Car1 position in relative to the Car0
        car1_pos = Pose()
        car1_pos.position.x_val= 0
        car1_pos.position.y_val= 0
        car1_pos.position.z_val= 2.73

        random_orientation = random.randint(-90, 90)
        car1_pos.orientation = airsim.to_quaternion(0 , 0,  random_orientation)
        client.simSetObjectPose("Car1", car1_pos)


        #internal camera settings
        self.angleh = self.initial_angleh
        self.anglev = self.initial_anglev
        self.fov = self.initial_fov

        #set the correct camera_pos
        self.camera_pos = Pose()
        self.camera_pos.position.x_val= self.camera_x
        self.camera_pos.position.y_val= self.camera_y
        self.camera_pos.position.z_val= self.camera_height

        tilt = self.anglev/57.2958
        pan = self.angleh/57.2958

        self.camera_pos.orientation = airsim.to_quaternion(tilt , 0,  pan)

        success = client.simSetCameraPose(self.camera_name, self.camera_pos, vehicle_name = self.vehicle_name)
        client.simSetCameraFov(self.camera_name, self.initial_fov, vehicle_name=self.vehicle_name)

        #get the exact position of camera from the engine
        car0_pos = client.simGetObjectPose("Car0").position;
        self.camera_pos_pix = client.simGetCameraInfo(self.camera_name, self.vehicle_name).pose.position + car0_pos

    def set_camera(self, fov=90, angleh = 0.0, anglev = 0.0):

        tilt = anglev/57.2958
        pan = angleh/57.2958
        self.camera_pos.orientation = airsim.to_quaternion(tilt , 0,  pan)

        self.client.simSetCameraPose(self.camera_name, self.camera_pos, vehicle_name = self.vehicle_name)
        self.client.simSetCameraFov(self.camera_name, fov, vehicle_name=self.vehicle_name)


    def reset(self):
        self.steps = 0
        self.outside_car = False


        self.initializeEnv()

        #the next state
        image_data = self.get_image()
        image = Image.frombytes('RGB', (image_data.width, image_data.height), image_data.image_data_uint8, 'raw', 'RGB', 0, 1)

        if self.screen_height!=image_data.width:
            image = image.resize((self.screen_height,self.screen_width ), resample=2)

        image = np.array(image)
        state = image
        return state


    def convert_rgb_to_gray(self, observation):
        r, g, b = observation[:, :, 0], observation[:, :, 1], observation[:, :, 2]
        observation = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return observation

    def step(self, action):

        # Initialize next state, reward, done flag
        self.next_state = None
        self.reward = None
        self.done = False
        self.steps += 1

        self.reward_x = 0
        self.reward_y = 0
        self.reward_object_size = 0

        try:
            self.reward, self.next_state = self.send_action(action)

        except Exception as e:
            self.reward = 0.0 #some reward for failure in the system
            self.next_state = self.past_image_state
            self.initializeEnv()
            print('Except in the step(self, action)')
            print(e)

        if self.steps==2000:
            self.done = True

        if self.outside_car == True:
            self.done = True

        return self.next_state, self.reward, self.done, {}


    def outside(self, Car1_pix):
        width = (self.captured_image/2.0)
        vx,vy = abs(Car1_pix[0]), abs(Car1_pix[1])

        if vx> width or vy>width:
            return True

        return False

    def calculate_object_detection_reward(self):
        reward = 0

        num_of_tries = 5

        for i in range(num_of_tries):
            car1 = self.client_det.simGetDetections(camera_name=self.camera_name, image_type=0)
            self.curr_detections_engine = car1
            if car1:
                car1 = car1[0]
                x1 = int(car1.box2D.min.x_val)
                y1 = int(car1.box2D.min.y_val)
                x2 = int(car1.box2D.max.x_val)
                y2 = int(car1.box2D.max.y_val)

                x_center = (x1+x2)/2.0
                y_center = (y1+y2)/2.0

                width = self.screen_height/2.0

                x_distance = abs(width - x_center)
                y_distance = abs(width - y_center)


                self.reward_x = (width - x_distance)/width
                self.reward_y = (width - y_distance)/width
                size_of_box = (y2-y1)*(x2-x1)/(self.screen_height*self.screen_height)

                self.reward_object_size = size_of_box

                reward = self.reward_object_size*self.reward_x*self.reward_y

                if x1==0 or y1==0:
                    reward = reward*0.3

                if x2==0 or y2==0:
                    reward = reward*0.3

                if x1==self.screen_height or y1==self.screen_height:
                    reward = reward*0.3

                if x2==self.screen_height or y2==self.screen_height:
                    reward = reward*0.3

                return reward

        self.outside_car = True
        reward = -10

        return reward

    def draw_bounding_box(self):

        response = self.curr_captured_cv_image
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
        img_rgb = img1d.reshape(response.height, response.width, 3)

        cars = self.curr_detections_engine

        if cars:
            car = cars[0]

            x1 = int(car.box2D.min.x_val)
            y1 = int(car.box2D.min.y_val)
            x2 = int(car.box2D.max.x_val)
            y2 = int(car.box2D.max.y_val)

            cv2.rectangle(img_rgb,(x1,y1),(x2,y2),(255,0,0),2)

            print('bounding: (x1,y1,x2,y2): ', (x1,y1,x2,y2))
        return img_rgb

    def get_bounding_box_engine(self):
        cars = self.curr_detections_engine

        if not cars:
            cars = self.client_det.simGetDetections(camera_name=self.camera_name, image_type=0)

        if cars:
            car = cars[0]

            x1 = int(car.box2D.min.x_val)
            y1 = int(car.box2D.min.y_val)
            x2 = int(car.box2D.max.x_val)
            y2 = int(car.box2D.max.y_val)

            return x1, y1, x2, y2

        return -1, -1, -1, -1

    def clamp(self, num, min_value, max_value):
        return max(min(num, max_value), min_value)

    def send_action(self,action):

        self.modified_cam = True

        local_reward = 0

        pan_a = action[0]
        tilt_a = action[1]
        zoom_a = action[2]

        local_reward = 0

        if pan_a==0:
            self.angleh+=self.delta
        elif pan_a==1:
            self.angleh-=self.delta

        if tilt_a==0:
            self.anglev+=self.delta
        elif tilt_a==1:
            self.anglev-=self.delta

        if zoom_a==0:
            self.fov+=1.0
        elif zoom_a==1:
            self.fov-=1.0

        #send the action to the camera
        self.fov = self.clamp(self.fov, 5, 90)
        self.anglev = self.clamp(self.anglev, -120, 120)
        self.angleh = self.clamp(self.angleh, -120, 120)
        #send the action to the camera
        t1 = time.time()
        self.set_camera(fov=self.fov, angleh = self.angleh, anglev = self.anglev)
        t2 = time.time()
        #print('Time to send camera action:', t2-t1)

        reward = 0

        #calculate the state
        try:

            t1 = time.time()
            image_data = self.get_image()
            image = Image.frombytes('RGB', (image_data.width, image_data.height), image_data.image_data_uint8, 'raw', 'RGB', 0, 1)
            if self.screen_height!=image_data.width:
                image = image.resize((self.screen_height,self.screen_width), resample=2)

            image = np.array(image)
            state = image

            t1 = time.time()
            reward = self.calculate_object_detection_reward()
            t2 = time.time()

            #keep track of successful state
            self.past_image_state = state

        except Exception as e:
            state = self.past_image_state
            print('Except in: send_action(self,action)')
            print(e)

        return reward, state


    def get_image(self):
        responses = self.client.simGetImages([airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene, False, False)],self.vehicle_name)  #scene vision image in uncompressed RGB array
        response = responses[0]

        self.curr_captured_cv_image = response
        return response

    def get_image2(self):
        responses = self.client.simGetImages([airsim.ImageRequest(self.camera_name2, airsim.ImageType.Scene, False, False)],self.vehicle_name)  #scene vision image in uncompressed RGB array
        response = responses[0]
        return response

    def get_reward_image(self):
        responses = self.client_reward.simGetImages([airsim.ImageRequest(self.camera_name, airsim.ImageType.Segmentation, False, False)],self.vehicle_name)  #Scene segmentation image
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
        img_rgb = img1d.reshape(response.height, response.width, 3)
        #print(response.height, response.width)

        return img_rgb

    def visualize_response_CV2(self,response):
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshheightape(response.height, response.width, 3)
        cv2.imshow(self.vehicle_name, img_rgb)

        cv2.waitKey(1)

    def display_CV2(self, x1, y1, x2, y2, next_state):
        x1= int(x1)
        y1=int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cv2.rectangle(next_state,(x1,y2),(x2,y1),(0,255,0),3)
        cv2.imshow('Display', next_state)
        cv2.waitKey(1)

def display_detection(x1, y1, x2, y2, next_state):
    x1= int(x1)
    y1=int(y1)
    x2 = int(x2)
    y2 = int(y2)
    cv2.rectangle(next_state,(x1,y2),(x2,y1),(0,255,0),3)
    display_observation(next_state)

#import sort algorithm with Kalman
from sort2 import *

def initialise_kalman(next_state):

    results = model(next_state)
    coord = results.xyxy[0].cpu().detach().numpy()

    #initialise Kalman tracker
    for i in range(len(coord)):
        if coord.shape[0]>0 and (int(coord[i, 5]) in ids_obj):

            x1 = coord[i,0]
            x2 = coord[i,2]

            y1 = coord[i,1]
            y2 = coord[i,3]

            conf = coord[i, 4]
            obj_class = 2

            box = [x1,y1,x2,y2,conf, obj_class]
            kf = KalmanBoxTracker(box)

            print('Iniliaze kalman done')

            return kf

    return kf



import pickle

def save_pickle(time_taken, total_rewards, local_steps_mean, rewards_x, rewards_y, black_rewards):
    time_taken2 = np.array(time_taken)
    total_rewards2 = np.array(total_rewards)
    local_steps_mean2 = np.array(local_steps_mean)
    rewards_x2 = np.array(rewards_x)
    rewards_y2 = np.array(rewards_y)
    black_rewards2 = np.array(black_rewards)
    data = [time_taken, total_rewards, local_steps_mean, rewards_x, rewards_y, black_rewards]

    with open(save_name, 'wb') as handle:
        pickle.dump(data, handle)

############################################### GET ACTION Flexible ###############
def get_action_tuner(next_state, kf, env, tol_width, no_zoom, max_zoom_size):

    action = [2, 2, 2]

    results = model(next_state)
    coord = results.xyxy[0].cpu().detach().numpy()
    try:
        for i in range(len(coord)):
            #print(coord[i,5])

            if coord.shape[0]>0 and (int(coord[i, 5]) in ids_obj):
                x1 = coord[i,0]
                x2 = coord[i,2]

                y1 = coord[i,1]
                y2 = coord[i,3]

                conf = coord[i, 4]
                obj_class = 2

                box = [x1,y1,x2,y2,conf, obj_class]
                kf.update(box)

                break
    except:
        print('No object detected by YOLO')

    predict_box = kf.predict()
    predict_box = predict_box[0]

    x1 = predict_box[0]
    x2 = predict_box[2]
    y1 = predict_box[1]
    y2 = predict_box[3]

    #print(x1, x2, y1, y2, conf, obj_class)
    x_center = (x1+x2)/2.0
    y_center = (y1+y2)/2.0

    area_shape = abs(x1-x2)*abs(y1-y2)

    dis_x = (x_center) - (env.screen_width)/2.0
    dis_y = (y_center) - (env.screen_height)/2.0

    #action directions
    pan_pos_1 = 0
    pan_neg_1 = 1

    tilt_pos_1 = 1
    tilt_neg_1 = 0

    #% of the screen_width
    tol_dis = (env.screen_width)*tol_width

    if abs(dis_x)<tol_dis:
        pan_action = 2

    elif dis_x>0:
        pan_action = pan_pos_1
    else:
        pan_action = pan_neg_1

    if abs(dis_y)<tol_dis:
        tilt_action =2

    elif dis_y>0:
        tilt_action = tilt_pos_1

    else:
        tilt_action = tilt_neg_1


    #get zoom action
    around_corner = False
    zoom_action = 2

    if x1==0 or y1==0:
        around_corner = True

    if x2==0 or y2==0:
        around_corner = True

    if x1==env.screen_height or y1==env.screen_height:
        around_corner = True

    if x2==env.screen_height or y2==env.screen_height:
        around_corner = True

    area_shape = abs(x1-x2)*abs(y1-y2)
    relative_size = area_shape/(env.screen_height*env.screen_height)

    if around_corner:
        zoom_action = 0

    elif relative_size<no_zoom:
        zoom_action = 1

    elif relative_size>no_zoom and relative_size<max_zoom_size:
        zoom_action = 2

    else:
        zoom_action = 0

    action = [pan_action, tilt_action, zoom_action]

    return action

##################################################################


##################################################################

Actions_Taken = [] #stores actions taken by camera
total_rewards = []
time_taken = []

rewards_x = []
rewards_y = []
black_rewards = []
local_steps_mean = []

env = PTZEnv()

random.seed(0)

for i in range(no_of_episodes):

    done = False
    start_time = time.time()

    initialized = False
    init_count = 0

    while (not initialized) and init_count <10:
        try:
            init_count+=1

            next_state = env.reset()
            kf = initialise_kalman(next_state)
            initialized = True

        except Exception as e:
            print('Failed to initialize kalman')
            print(e)

    local_steps = 0
    reward = 0

    time_to_sleep_corrected = 0
    first_time = 10
    sleeping_times = []

    while not done:
        local_steps +=1

        t1 = time.time()
        #pass the thresholds to the rule-based controlled to move PTZ camera.
        action = get_action_tuner(next_state, kf, env, tol_width = 0.10, no_zoom = 0.135, max_zoom_size=0.145)

        if time_to_sleep_corrected>0.0 and first_time<=0:
            time.sleep(time_to_sleep_corrected/1000)

        next_state, rew, done, _ = env.step(action)
        t2 = time.time()

        #Make the delay within 25 - 30ms. This assumes machine can run simulation faster.
        if first_time>0:
            first_time-=1
            curr_time_to_sleep_corrected = max(0.0, (30.0 - (t2-t1)*1000.0))
            sleeping_times.append(curr_time_to_sleep_corrected)
            time_to_sleep_corrected = np.mean(np.array(sleeping_times))


        reward_x, reward_y, black_reward = env.reward_x, env.reward_y, env.reward_object_size

        time_taken.append(t2-t1)
        reward = reward + rew
        rewards_x.append(reward_x)
        rewards_y.append(reward_y)
        black_rewards.append(black_reward)

        if done:

            #ignore local_steps<100. This can happen when simulator is resetting the car, and it get strucks
            if local_steps>100:
                total_rewards.append(reward)
                local_steps_mean.append(local_steps)

                print(i, local_steps,' : done-----reward:',int(reward), ' time taken:', int(sum(time_taken)*1000.0/len(time_taken)), ' Avg Reward:', int(sum(total_rewards)/len(total_rewards)),
                '  Steps:', int(sum(local_steps_mean)/len(local_steps_mean)), ' rewards_x:', round(sum(rewards_x)/len(rewards_x), 2),
                '  rewards_y:', round(sum(rewards_y)/len(rewards_y), 2), '  size:', round(sum(black_rewards)/len(black_rewards), 2))

                save_pickle(time_taken, total_rewards, local_steps_mean, rewards_x, rewards_y, black_rewards)

            else:
                print('Done at:', local_steps)


            next_state = env.reset()


#delay_per_step
time_taken = np.array(time_taken)
print('Avg Time taken:', np.mean(time_taken))

total_rewards = np.array(total_rewards)
print('Average Reward is:', np.mean(total_rewards))

local_steps_mean = np.array(local_steps_mean)
print('local_steps_mean is:', np.mean(local_steps_mean))

rewards_x = np.array(rewards_x)
rewards_y = np.array(rewards_y)
black_rewards = np.array(black_rewards)
print('rewards_x mean is:', np.mean(rewards_x))
print('rewards_y mean is:', np.mean(rewards_y))
print('black_rewards mean is:', np.mean(black_rewards))

import pickle
data = [time_taken, total_rewards, local_steps_mean, rewards_x, rewards_y, black_rewards]

with open(save_name, 'wb') as handle:
    pickle.dump(data, handle)
