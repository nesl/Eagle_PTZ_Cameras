"""
Train Sc-4: Variable backgrounds + Trees + image augmentations + humans

Image augmentations are realized as python functions in this file.

OpenAI: GYM Abstractions which are part of the Scene-Controller exposing Eagle Capbilitis to train
policies for Sc-4 scenes

Assumes: This file assumes that 6 parallel virtual worlds are running on the below ports:
[41451, 41452, 41453, 41454, 41455, 41456]

if you want to experiment on single or few parallel worlds,
change the value: 'num_envs = 6' in the main function in the end to a desired value: eg: 'num_envs = 1'
for one virtual world running at 41451

"""

#assign which cuda GPU to use
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tmp_path = "Trained_sc3/"

#print the python env being used
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


### These are the image augmentations implemented in python
def random_hue(x, saturation=None):
    if saturation is None:
        saturation = np.random.randint(10)

    x = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    v = x[:, :, 2]
    v = np.where(v <= 255 + saturation, v - saturation, 255)
    x[:, :, 2] = v
    x = cv2.cvtColor(x, cv2.COLOR_HSV2BGR)

    return x

def random_saturation(x, saturation=None):
    if saturation is None:
        saturation = np.random.randint(30)

    x = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    v = x[:, :, 2]
    v = np.where(v <= 255 - saturation, v - saturation, 255)
    x[:, :, 2] = v
    x = cv2.cvtColor(x, cv2.COLOR_HSV2BGR)

    return x

def random_brightness(x, brightness=None):
    if brightness is None:
        brightness = np.random.uniform(0.5, 2.5)

    hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255)
    x = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return x

def random_contrast(x, contrast=None):
    if contrast is None:
        contrast = np.random.randint(50)  # too large will be too dark

    x = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    x[:, :, 2] = [[max(pixel - contrast, 0) if pixel < 190 else min(
        pixel + contrast, 255)
                   for pixel in row] for row in x[:, :, 2]]
    x = cv2.cvtColor(x, cv2.COLOR_HSV2BGR)
    return x


def random_color(x):
    x = random_hue(x)
    x = random_saturation(x)
    x = random_brightness(x)
    return x

def shadow(x):
    H, W, _ = x.shape

    top_x, bot_x = 0, W
    top_y, bot_y = H * np.random.uniform(), H * np.random.uniform()

    hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    shadow_mask = 0 * hsv[:, :, 1]
    X_m, Y_m = np.mgrid[0:H, 0:W]

    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (
                Y_m - top_y) >= 0)] = 1

    shadow_density = .5
    left_side = shadow_mask == 1
    right_side = shadow_mask == 0

    if np.random.randint(2) == 1:
        hsv[:, :, 2][left_side] = hsv[:, :, 2][left_side] * shadow_density
    else:
        hsv[:, :, 2][right_side] = hsv[:, :, 2][right_side] * shadow_density

    x = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return x

def sharpen(x):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    x = cv2.filter2D(x, -1, kernel)
    return x


def salt_and_pepper(x, p=0.5, a=0.09):
    noisy = x.copy()

    # salt
    num_salt = np.ceil(a * x.size * p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in x.shape]
    noisy[tuple(coords)] = 1

    # pepper
    num_pepper = np.ceil(a * x.size * (1. - p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in x.shape]
    noisy[tuple(coords)] = 0

    return noisy

def do_randomization(observation ,p=0.90):

     if np.random.uniform() > p:
         observation = random_color(observation)

     if np.random.uniform() > p:
         observation = shadow(observation)

     if np.random.uniform() > p:
        observation = shadow(observation)

     if np.random.uniform() > p:
         observation = sharpen(observation)

     if np.random.uniform() > p:
         observation = salt_and_pepper(observation)

     if np.random.uniform() > p:
         observation = salt_and_pepper(observation)

     if np.random.uniform() > p:
        observation = salt_and_pepper(observation)

     return observation
##################################




#%matplotlib inline
def display_observation(image):
    import matplotlib.pyplot as plt
    imgplot = plt.imshow(image)
    plt.show()


class PTZEnv(gym.Env):

    #camera starting pose
    camera_height = -8.0

    #camera initial fov
    initial_fov = 60
    initial_angleh = 0
    initial_anglev = -10.0

    #state space size
    screen_height = 120
    screen_width = 120
    captured_image = 120
    seg_img = 120

    #camera is attached to Car0
    camera_name = "0"
    vehicle_name="Car0"

    camera_pos = None

    #store the current camera values
    angleh = initial_angleh
    anglev = initial_anglev
    fov = initial_fov

    #change in pan,tilt.
    delta = 2.0
    #penalty on changing cameras PTZ
    penalty_movement = 0.01

    curr_captured_cv_image = None
    curr_detections_engine = None

    def __init__(self, start_airsim=True, env_id = 0):

        # actions -> PTZ
        # actions -> PTZ, 3 for pan, 3 for tilt, 3 for zoom
        self.action_space = spaces.MultiDiscrete([ 3, 3, 3])

        # given image from Airsim simulator
        self.observation_space = spaces.Box(low=0, high=255,shape=(self.screen_height, self.screen_width, 1), dtype=np.uint8)

        self.steps = 0

        #records if camera was modified
        self.modified_cam = False

        #used to store one past state to be used when image collection fails
        self.past_image_state = None

        #if car is outside of boundary
        self.outside_car = False

        self.port = 41451+env_id

        if start_airsim:
            self.client = airsim.CarClient(port = self.port)
            self.initializeEnv()
            self.client_det = airsim.CarClient(port = self.port)
            self.client_det.simAddDetectionFilterMeshName(camera_name=self.camera_name, image_type=0, mesh_name="Car1",vehicle_name = self.vehicle_name)

        self.reward_x = 0
        self.reward_y = 0
        self.reward_object_size = 0
        self.curr_seg_image = None

        #keep track of elapsed time in each step
        self.curr_time = 0 #used to maintain the tick-rate

        #set this this appropritate value to make training step delay within in 25ms - 30ms
        #this will depend on the compute of your machine. See the FPS of stable_baselines3 during training
        self.delay_to_introduce = 0.0/1000.0

        #the most recent current delay in the pipeline
        self.curr_delay=0

    def clamp(self, num, min_value, max_value):
        return max(min(num, max_value), min_value)

    def initializeEnv(self):

        client = self.client

        client.reset()

        #Random Iniliaze the Car1 position in relative to the Car0
        car1_pos = Pose()
        car1_pos.position.x_val= random.randint(-5, 5)
        car1_pos.position.y_val= random.randint(-5, 5)
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
        self.camera_pos.position.x_val= 0
        self.camera_pos.position.y_val= 0
        self.camera_pos.position.z_val= self.camera_height

        tilt = self.anglev/57.2958
        pan = self.angleh/57.2958

        self.camera_pos.orientation = airsim.to_quaternion(tilt , 0,  pan)

        success = client.simSetCameraPose(self.camera_name, self.camera_pos, vehicle_name = self.vehicle_name)

        fov_to_set = random.randint(40, 60)
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
        state = image_gray

        return state


    def convert_rgb_to_gray(self, observation):
        r, g, b = observation[:, :, 0], observation[:, :, 1], observation[:, :, 2]
        observation = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return observation

    def step(self, action):

        self.curr_time = time.time()

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
            #send action to the camera
            self.reward, self.next_state, modified_cam = self.send_action(action)

        except Exception as e:
            self.reward = 0.0 #some reward for failure in the system
            self.next_state = self.past_image_state
            self.initializeEnv()
            print('Except in the step(self, action)')
            print(e)

        if self.steps==2000:
            self.done = True
            print('completed episode:2000')

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

    def calculate_object_detection_reward(self):
        reward = 0

        num_of_tries = 5

        #this is needed, as sometimes the engine may fail even when Car is in FOV
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

        local_reward = 0

        pan_a = action[0]
        tilt_a = action[1]
        zoom_a = action[2]

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

        if self.delay_to_introduce >0.0:
            time.sleep(self.delay_to_introduce)

        self.set_camera(fov=self.fov, angleh = self.angleh, anglev = self.anglev)

        reward = 0

        #calculate the state
        try:
            image_data = self.get_image()
            image = Image.frombytes('RGB', (image_data.width, image_data.height), image_data.image_data_uint8, 'raw', 'RGB', 0, 1)
            if self.screen_height!=image_data.width:
                image = image.resize((self.screen_height,self.screen_width), resample=2)

            image = np.array(image)
            image = do_randomization(image)

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

    def get_reward_image(self):
        responses = self.client_reward.simGetImages([airsim.ImageRequest(self.camera_name, airsim.ImageType.Segmentation, False, False)],self.vehicle_name)  #Scene segmentation image
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
        img_rgb = img1d.reshape(response.height, response.width, 3)

        return img_rgb

    def visualize_response_CV2(self,response):
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshheightape(response.height, response.width, 3)
        cv2.imshow(self.vehicle_name, img_rgb)

        cv2.waitKey(1)



# ################################### PPO Training code
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model_color')
        self.save_path_recent = os.path.join(log_dir, 'recent_model_color')

        self.best_mean_reward = -np.inf

        self.curr_itr = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          self.curr_itr+=1
          if len(x) > 0:

              self.model.save(self.save_path_recent)

              if self.curr_itr%50==0:
                  reg_path = self.save_path_recent+str(self.curr_itr)
                  self.model.save(reg_path)

              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True


#custom network architecture
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN_pi(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN_pi, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
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
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs_pi = dict(
    features_extractor_class=CustomCNN_pi,
    features_extractor_kwargs=dict(features_dim=64),
)


from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv,VecMonitor
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env(env_id):
    def _init():
        env = PTZEnv(env_id = env_id)
        return env
    return _init

if __name__ == '__main__':
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv"])
    num_envs = 6

    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    env = VecMonitor(env, tmp_path)

    model = PPO(policy = "CnnPolicy", env=env, verbose=1, n_steps=1024*4, batch_size=256, clip_range=0.2, policy_kwargs = policy_kwargs_pi)
    model.set_logger(new_logger)

    callback = SaveOnBestTrainingRewardCallback(check_freq=1024*4, log_dir=tmp_path)
    model.learn(total_timesteps=2500000000, callback=callback)
