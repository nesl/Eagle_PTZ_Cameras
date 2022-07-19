
"""
Evaluate the trained policies
Sample checkpoint is provided.

# Replace with your checkpoint
checkpoint1 = 'Trained_dt/dt_model.zip'

The pickle structure is available in this file
Results are also visually displayed

During evaluation, we fix the sensing to actuation delay within 25 - 30ms at line: 643
This assumes machine can run simulation faster.


## Selectively track human or car:
#0 is for car, 1 is for humans
to_track = 0

"""

# set the cuda GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

global_port = 41451

no_of_episodes = 100
image_size_to_use = 120
enable_debug = False

#0 is for car, 1 is for humans
to_track = 0

checkpoint1 = 'Trained_dt/dt_trained.zip'

save_name = 'evaluation.pickle'

print('Running on port:', global_port)
print(save_name)

checkpoint_list = [checkpoint1]

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

random.seed(0)

#%matplotlib inline
def display_observation(image):
    import matplotlib.pyplot as plt
    imgplot = plt.imshow(image)
    plt.show()

class PTZEnv(gym.Env):

    #camera starting pose
    camera_height = -8.0
    camera_x = 0
    camera_y = 0

    #camera initial fov
    initial_fov = 40
    initial_angleh = 0
    initial_anglev = -20.0

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

    curr_captured_cv_image = None
    curr_detections_engine = None

    def __init__(self, start_airsim=True, env_id = 0):

        # actions -> PTZ
        # actions -> PTZ, 3 for pan, 3 for tilt, 3 for zoom
        self.action_space = spaces.MultiDiscrete([ 3, 3, 3]) #spaces.Discrete(11)

        # given image from Airsim simulator
        high = np.array([1.0])
        low = np.array([0.0])

        self.observation_space = spaces.Dict({'image':spaces.Box(low=0, high=255,shape=(self.screen_height, self.screen_width, 1), dtype=np.uint8),
                                             'vector':spaces.Box(low, high, dtype=np.float32)})

        self.steps = 0

        #records if camera was modified
        self.modified_cam = False

        #used to store one past state to be used when image collection fails
        self.past_image_state = None

        #if car is outside of boundary
        self.outside_car = False

        self.port = global_port

        if start_airsim:
            self.client = airsim.CarClient(port = self.port)
            self.initializeEnv()

            self.client_det = airsim.CarClient(port = self.port)
            #self.client_det.simSetDetectionFilterRadius(camera_name=self.camera_name, image_type=0, radius_cm =100 * 100, vehicle_name = self.vehicle_name)

            if to_track==0:
                self.client_det.simAddDetectionFilterMeshName(camera_name=self.camera_name, image_type=0, mesh_name="Car1",vehicle_name = self.vehicle_name)

            else:
                self.client_det.simAddDetectionFilterMeshName(camera_name=self.camera_name, image_type=0, mesh_name='HumanCharacter*',vehicle_name = self.vehicle_name)

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

    def clamp(self, num, min_value, max_value):
        return max(min(num, max_value), min_value)

    def initializeEnv(self):

        client = self.client

        client.reset()

        #Iniliaze the Car1 position in relative to the Car0
        car1_pos = Pose()
        car1_pos.position.x_val= 0 #random.randint(-5, 5)
        car1_pos.position.y_val= 0 #random.randint(-5, 5)
        car1_pos.position.z_val= 0

        if to_track==1:
            car1_pos.position.x_val= 10
            car1_pos.position.y_val= 10
            car1_pos.position.z_val= 0

        random_orientation = random.randint(0, 360)*0.0174533
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

        #fov_to_set = random.randint(40, 60)

        fov_to_set = 40
        self.fov = fov_to_set
        client.simSetCameraFov(self.camera_name, fov_to_set, vehicle_name=self.vehicle_name)

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

        state = {'image':image_gray, 'vector':np.array([to_track])}

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
            #send action to the camera
            self.reward, self.next_state = self.send_action(action)

        except Exception as e:
            self.reward = 0.0 #some reward for failure in the system
            self.next_state = self.past_image_state
            self.initializeEnv()
            print('Except in the step(self, action)')
            print(e)

        if self.steps==2000:
            self.done = True
            #print('completed episode:2000')

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

        # to handle automatic resetting of human virtual world from simulator
        num_of_tries = 8

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
        print('done:',self.steps)

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


    def send_action(self,action):

        self.modified_cam = True

        local_reward = 0

        pan_a = action[0]
        tilt_a = action[1]
        zoom_a = action[2]

        #action = int(action)
        self.modified_cam = True
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

        self.set_camera(fov=self.fov, angleh = self.angleh, anglev = self.anglev)

        reward = 0

        #calculate the state
        try:

            #t1 = time.time()
            image_data = self.get_image()
            image = Image.frombytes('RGB', (image_data.width, image_data.height), image_data.image_data_uint8, 'raw', 'RGB', 0, 1)
            if self.screen_height!=image_data.width:
                image = image.resize((self.screen_height,self.screen_width), resample=2)

            image = np.array(image)

            image_gray = self.convert_rgb_to_gray(image)
            image_gray = image_gray.reshape(self.screen_height, self.screen_width, 1)

            state = {'image':image_gray, 'vector':np.array([to_track])}

            t1 = time.time()
            reward = self.calculate_object_detection_reward()
            t2 = time.time()
            #print('time to cal box reward:', t2-t1)

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



import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                n_input_channels = subspace.shape[0]
                print('n_input_channels:',n_input_channels)
                self.cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 64, kernel_size=5, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )

                # Compute shape by doing one forward pass
                with th.no_grad():
                    n_flatten = self.cnn(
                        th.as_tensor(subspace.sample()[None]).float()
                    ).shape[1]

                print('n_flatten:', n_flatten)
                extractors[key] = self.cnn
                total_concat_size += n_flatten

            elif key == "vector":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 2)
                total_concat_size += 2

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size
        print('self._features_dim:',self._features_dim)

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


policy_kwargs_custom = dict(
    features_extractor_class=CustomCombinedExtractor
)



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


from stable_baselines3.common.vec_env import DummyVecEnv
env = DummyVecEnv([lambda: PTZEnv()])

model = PPO(policy = "MultiInputPolicy", env=env, verbose=1, n_steps=1024*8, batch_size=256, clip_range=0.2, policy_kwargs=policy_kwargs_custom)

Actions_Taken = [] #stores actions taken by camera
total_rewards = []
time_taken = []


rewards_x = []
rewards_y = []
black_rewards = []
local_steps_mean = []

Captured_images = []

env=PTZEnv()

for checkpoint in checkpoint_list:
    model.set_parameters(checkpoint)
    print('model loaded:', checkpoint)

    # a dummy run
    obs = env.reset()
    model.predict(obs, deterministic=True)

    for i in range(no_of_episodes):#i < 5.0:
        done = False
        start_time = time.time()
        obs = env.reset()
        local_steps = 0
        reward = 0

        time_to_sleep_corrected = 0
        first_time = 10
        sleeping_times = []


        while not done:
            local_steps +=1

            t1 = time.time()

            action, _state = model.predict(obs, deterministic=True)

            if time_to_sleep_corrected>0.0 and first_time<=0:
                time.sleep(time_to_sleep_corrected/1000)

            obs, rew, done, _= env.step(action)
            t2 = time.time()

            #Make the delay within 25 - 30ms. This assumes machine can run simulation faster.
            if first_time>0:
                first_time-=1
                curr_time_to_sleep_corrected = max(0.0, (27.0 - (t2-t1)*1000.0))
                sleeping_times.append(curr_time_to_sleep_corrected)
                time_to_sleep_corrected = np.mean(np.array(sleeping_times))

            time_taken.append(t2-t1)
            reward = reward + rew

            reward_x = env.reward_x
            reward_y = env.reward_y
            black_reward = env.reward_object_size

            rewards_x.append(reward_x)
            rewards_y.append(reward_y)
            black_rewards.append(black_reward)

            Actions_Taken.append(action)

            if done:
                #ignore local_steps<200. hanlde issues with simulator resetting the objects
                if local_steps>200:
                    total_rewards.append(reward)
                    local_steps_mean.append(local_steps)

                    print(i, local_steps,' : done-----reward:',int(reward), ' time taken:', int(sum(time_taken)*1000.0/len(time_taken)), ' Avg Reward:', int(sum(total_rewards)/len(total_rewards)),
                    '  Steps:', int(sum(local_steps_mean)/len(local_steps_mean)), ' rewards_x:', round(sum(rewards_x)/len(rewards_x), 2),
                    '  rewards_y:', round(sum(rewards_y)/len(rewards_y), 2), '  size:', round(sum(black_rewards)/len(black_rewards), 2))

                    save_pickle(time_taken, total_rewards, local_steps_mean, rewards_x, rewards_y, black_rewards)

                else:
                    print('Done at:', local_steps)

                obs = env.reset()

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
