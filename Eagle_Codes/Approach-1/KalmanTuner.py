"""
Tune the Kalman parameters:
- We tune: kf.R and kf.P variables.
- Which is Kalman filter measurement noises and covariances.


This reference implementation uses the EagleSim bounding boxes to tune the parameters.
The Kalman implementation in Sort2.py is tuned using the tuned controller (ControllerTuner.py).


Mango discovered parameters:
best parameters: {'P_m1': 1002.6734402794423, 'R_m': 58.55371237129906, 'p_m2': 92.41972548772546}
"""

file_name = 'Kalman_tuner.pickle'

global_port = 41451
image_size_to_use = 240

########################################## Mango tuning details
from mango.tuner import Tuner
from scipy.stats import uniform

batch_size = 1
global_per_reward_episode = 10
mango_iterations = 100


#parameter space for kalman tuner
param_space = {
    'R_m': uniform(0, 100.0),
    'P_m1': uniform(100.0, 10000.0),
    'p_m2': uniform(0, 100.0)
}


conf = dict()
conf['batch_size'] = batch_size
conf['initial_random'] = batch_size
conf['num_iteration'] = mango_iterations
conf['domain_size'] = 5000



import sys
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
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


    max_reward = 200.0

    #store the current camera values

    angleh = initial_angleh
    anglev = initial_anglev
    fov = initial_fov
    delta = 2.0
    penalty_movement = 0.01


    curr_captured_cv_image = None
    curr_detections_engine = None

    def __init__(self, start_airsim=True, env_id = 0):

        # actions -> PTZ
        # actions -> PTZ, 3 for pan, 3 for tilt, 3 for zoom
        self.action_space = spaces.MultiDiscrete([ 3, 3, 3]) #spaces.Discrete(11)

        self.observation_space = spaces.Box(low=0, high=255,shape=(self.screen_height, self.screen_width, 1), dtype=np.uint8)
        self.steps =0

        #records if camera was modified
        self.modified_cam = False

        #used to store one past state to be used when image collection fails
        self.past_image_state = None

        #if car is outside of boundary
        self.outside_car = False

        self.port = global_port+env_id

        if start_airsim:
            self.client = airsim.CarClient(port = self.port)
            self.initializeEnv()
            self.client_det = airsim.CarClient(port = self.port)
            #self.client_det.simSetDetectionFilterRadius(camera_name=self.camera_name, image_type=0, radius_cm =100 * 100, vehicle_name = self.vehicle_name)
            self.client_det.simAddDetectionFilterMeshName(camera_name=self.camera_name, image_type=0, mesh_name="Car1",vehicle_name = self.vehicle_name)

        self.reward_x = 0
        self.reward_y = 0
        self.reward_object_size = 0
        self.curr_seg_image = None

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

        random_orientation = random.randint(-90, 90)/57.2958

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
        image_gray = self.convert_rgb_to_gray(image)
        image_gray = image_gray.reshape(self.screen_height, self.screen_width, 1)

        state = image_gray

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
        modified_cam  = False

        try:
            self.reward, self.next_state, modified_cam = self.send_action(action)

        except Exception as e:
            self.reward = 0.0 #some reward for failure in the system
            self.next_state = self.past_image_state
            self.initializeEnv()
            print('Except in the step(self, action)')
            print(e)

        if self.steps==2000:
            self.done = True
            #print('Done Steps:', 2000)

        if self.outside_car == True:
            self.done = True

        if modified_cam:
            self.reward = self.reward - self.penalty_movement

        return self.next_state, self.reward, self.done, {}

    def outside(self, Car1_pix):
        width = (self.captured_image/2.0)
        vx,vy = abs(Car1_pix[0]), abs(Car1_pix[1])

        if vx> width or vy>width:
            return True

        return False

    def calculate_segmentation_reward(self):
        seg_image = self.get_reward_image()
        seg_image = self.convert_rgb_to_gray(seg_image)
        self.curr_seg_image = seg_image

        v = np.where(seg_image==0, 0, 1)
        self.reward_object_size = float(np.sum(v))/(self.seg_img*self.seg_img*0.1)

        reward = 0

        if self.reward_object_size==0:
                self.outside_car = True
                reward = -10.0

        else:
            result = np.where(v == 1)
            width = self.seg_img/2.0
            sum_x = abs(np.mean(result[0]) - width)
            sum_y = abs(np.mean(result[1]) - width)

            #print('sum_x, sum_y:', sum_x, sum_y)
            self.reward_x = (width - sum_x)/width
            self.reward_y = (width - sum_y)/width

            reward = self.reward_object_size*self.reward_x*self.reward_y


        return reward

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

                #print('box:', x1, x2, y1, y2)
                x_center = (x1+x2)/2.0
                y_center = (y1+y2)/2.0
                #print('box:', x_center, y_center)

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

        #action = int(action)
        self.modified_cam = False
        local_reward = 0

        if pan_a==0:
            self.angleh+=self.delta
            self.modified_cam = True

        elif pan_a==1:
            self.angleh-=self.delta
            self.modified_cam = True

        if tilt_a==0:
            self.anglev+=self.delta
            self.modified_cam = True

        elif tilt_a==1:
            self.anglev-=self.delta
            self.modified_cam = True

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

        reward = 0

        #calculate the state
        try:

            t1 = time.time()
            image_data = self.get_image()
            image = Image.frombytes('RGB', (image_data.width, image_data.height), image_data.image_data_uint8, 'raw', 'RGB', 0, 1)
            if self.screen_height!=image_data.width:
                image = image.resize((self.screen_height,self.screen_width), resample=2)

            image = np.array(image)
            image_gray = self.convert_rgb_to_gray(image)
            image_gray = image_gray.reshape(self.screen_height, self.screen_width, 1)

            state = image_gray

            t1 = time.time()
            reward = self.calculate_object_detection_reward()
            t2 = time.time()

            #keep track of successful state
            self.past_image_state = state

        except Exception as e:
            state = self.past_image_state
            print('Except in: send_action(self,action)')
            print(e)

        return reward, state, self.modified_cam


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


#import algorithm with Kalman
from sort2 import *

def initialise_kalman(next_state, env_id, R_m, P_m1, p_m2):
    global envs, model

    env = envs[env_id]
    x1, y1, x2, y2 = env.get_bounding_box_engine()
    conf = 1.0
    obj_class = 1
    box = [x1,y1,x2,y2,conf, obj_class]
    kf = KalmanBoxTracker(box, R_m, P_m1, p_m2)

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
def get_action_tuner(next_state, kf, env, tol_width = 0.15, no_zoom = 0.25, max_zoom_size=0.40):
    action = [2, 2, 2]
    x1, y1, x2, y2 = env.get_bounding_box_engine()

    conf = 1.0
    obj_class = 1

    box = [x1,y1,x2,y2,conf, obj_class]

    try:
        kf.update(box)

    except:
        print('Error in kf.update')
        return action

    predict_box = kf.predict()
    predict_box = predict_box[0]

    x1 = predict_box[0]
    x2 = predict_box[2]
    y1 = predict_box[1]
    y2 = predict_box[3]

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

##################---------Tracking objective of reward----------##########
def get_objective(env_id = 0, R_m=10.0, P_m1=1000.0, p_m2=10.0):
    global envs, global_per_reward_episode

    no_of_episodes = global_per_reward_episode
    total_obj_reward = 0.0
    local_steps = 0

    time_taken = []

    env = envs[env_id]



    for i in range(no_of_episodes):
        done = False
        start_time = time.time()

        initialized = False
        init_count = 0

        first_time = True
        time_to_sleep_corrected = 0

        while (not initialized) and init_count <10:
            try:
                init_count+=1

                next_state = env.reset()
                kf = initialise_kalman(next_state, env_id, R_m, P_m1, p_m2)
                initialized = True

            except Exception as e:
                print('Failed to initialize kalman')
                print(e)


        reward = 0

        while not done:

            local_steps +=1

            t1 = time.time()

            #Best rule based controller parameters for perfect-boxes from Mango
            action = get_action_tuner(next_state, kf, env, tol_width=0.1, no_zoom=0.38, max_zoom_size=0.38)


            #to make FPS equal to the 33, or 30 ms delay
            if time_to_sleep_corrected>0:
                time.sleep(time_to_sleep_corrected/1000)

            next_state, rew, done, _ = env.step(action)
            t2 = time.time()

            if first_time:
                first_time = False
                time_to_sleep_corrected = max(0, (30.0 - (t2-t1)*1000))

            time_taken.append(t2-t1)
            reward = reward + rew

            if done:
                print(i, local_steps,' : done-----reward:',reward, ' time taken:', int(sum(time_taken)*1000.0/len(time_taken)))
                total_obj_reward+=reward
                next_state = env.reset()

    print(int(R_m), int(P_m1), int(p_m2), ': Reward:', int(total_obj_reward/no_of_episodes), ' steps: ', int(local_steps/(no_of_episodes*20)),  ' time taken:', int(sum(time_taken)*1000.0/len(time_taken)))

    return total_obj_reward/no_of_episodes


def objfunc(args_list):
    results = []

    for i in range(batch_size):
        hyper_par = args_list[i]
        R_m = hyper_par['R_m']
        P_m1 = hyper_par['P_m1']
        p_m2 = hyper_par['p_m2']
        result = get_objective(i, R_m, P_m1, p_m2)
        results.append(result)

        save_pickle_hyper(hyper_par, result)

    return results


def save_pickle_hyper(hyper_pars, reward):

    #try loading the data first
    data = []
    try:
        with open(file_name, 'rb') as handle:
            data = pickle.load(handle)
    except:
        print('No file named:', file_name)

    #save the data next
    data.append([hyper_pars, reward])

    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle)


#initiliaze Mango first
data = []
try:
    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)

    print('Initializing Mango from past runs')
    hyper_pars = []
    objectives = []
    for d in data:
        h = d[0]
        r = d[1]
        hyper_pars.append(h)
        objectives.append(r)

    conf['initial_custom'] = {'hyper_pars':hyper_pars, 'objectives':objectives}

except:
    #print('mango not initialized')
    pass


envs = []
for i in range(batch_size):
    envs.append(PTZEnv(env_id=i))


random.seed(0)

tuner = Tuner(param_space, objfunc,conf)

print('*'*100)
print('Calling Tuner maximize')

results = tuner.maximize()
print('best parameters:', results['best_params'])
print('best accuracy:', results['best_objective'])
