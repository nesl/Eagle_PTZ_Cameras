# Eagle: End-to-end Deep Reinforcement Learning based Autonomous Control of PTZ Cameras

We provide **EagleSim** simulator and Eagle source codes.  The *requirements.txt* contains the different libraries required.


## 1. EagleSim Simulator
There are two different ways to use EagleSim.
 1. Using pre-compiled binaries: This is an easy way to set up the PTZ camera abstractions and get going.
 We provide pre-compiled binaries for vehicle tracking and human tracking scenarios.
 The pre-compiled binaries are tested on Ubuntu 18 and Ubuntu 20. If this doesn't work for you, we recommend compiling the binaries using source code for your desired operating system configuration.


 2. Compiling the source code. We provide the source code of the virtual world built over Unreal engine that creates vehicle tracking and human tracking scenarios. This source code can be used to compile the desired binaries.


### 1.1 Pre-compiled Binaries

The binaries are available at the following [link](https://drive.google.com/drive/folders/1bSXGXncTOqWH2KirhMq8_iCwEEPqxJTC?usp=sharing).

1. Vehicle_Tracking_Binary: This is used to create vehicle tracking scenarios (Sc-1 to Sc-5) with different tracking complexity.
2. Human_Tracking_Binary: This is used to create human tracking scenarios. We use it to enable dynamic tasking of Eagle policies.
3. Vehicle_Tracking_Binary_More_Vehicles: This provides a large set of vehicles which can be used to create more tracking scenarios.


#### Running a Binary
Download the binary locally. Put the *settings_sample.json* to the appropriate folder used locally for settings of the AirSim simulator. The folder location is generally /HomeFolder/Documents/AirSim in Linux machines. Rename the file to settings.json, which the EagleSim will read. The settings.json is an extension of the AirSim settings file with more options to enable different training scenarios. The scenes can be started on different ports by modifying the "ApiServerPort" in settings.json, and the binaries are started from different terminals to start parallel scenes. The code uses the ports [41451, 41452, 41453, 41454, 41455, 41456] to run.

Go to the local directory:
```
cd PATH_TO_BINARY/LinuxNoEditor/
```


Start the binary using the following command:

```
./Blank_car.sh -WINDOWED -opengl4
```


### 1.2 Controlling Scene Complexity
Scene complexity is controlled by enabling different components in the *settings_sample.json* as shown below. These scenes are enabled in the custom written Unreal engine source code provided for the virtual worlds.

```
"EnableBoundary": "yes",
"EnableMaterials":"yes",
"EnableOutsideTrees":"yes",
"EnableInsideTrees":"yes",
"EnableHumans": "yes",
```
*EnableBoundary* when selected to "yes", randomly selects a boundary out of 25 different materials, and also modifies boundary over-time.
*EnableMaterialls* when selected to "yes", randomly selects a floor material out of 25 different materials, and also modifies floor material over-time.
*EnableOutsideTrees*  and *EnableInsideTrees* when selected to "yes" randomly place different trees in the scene, which are also modified with time.
*EnableHumans* places different human characters in the scenes. For vehicle tracking binaries, multiple human characters are present which are moving
on random paths, and also change their mesh characters. In the case of human tracking binary, only one human is present in the scene, and does animated walking on random paths. Human character will also change its mesh apperance to 4 different characters in the human tracking binary.

### 1.3 Number of Vehicles in the Scene
The vehicles are spawned by extending the AirSim *setting.json* file. More details are [here](https://microsoft.github.io/AirSim/settings/). We provide the sample *settings_sample.json*, and include settings used for different scenes in Eagle scenarios in their respective folders.
*Vehicle_Tracking_Binary* and *Vehicle_Tracking_Binary_More_Vehicles* support a large collection of vehicles as shown in the included *settings_sample.json* file.

### 1.4 Controlling Vehicles in the Scene
The vehicles in the scenes are controlled using AirSim API. The code to control the vehicle on random paths is present in file *ControlCar.py*.

### 1.5 Controlling Humans in the Scene
The code to control humans is custom written and is provided in the source codes for the virtual worlds. The humans are enabled to move on random paths
within the virtual worlds limits.

### 1.6 PTZ Camera Placement
We use the camera of the *Car0* as the PTZ camera. The location of the vehicle and its camera is controllable in the *settings_sample.json* file.

### 1.7 PTZ Camera View
The PTZ camera view can be visualized using:

```
python ViewCamera.py
```

### 1.8 PTZ Abstraction: Control Pan-Tilt-Zoom Parameters
The abstractions are exposed via Python wrappers.
The camera controller supports real-time change of PTZ parameters.
Object trackers tracks the object of interest and provides its bounding-boxes.
Overall the scene control of PTZ is enabled using the Open-AI Gym abstractions.
A sample scene control in Open-AI GYM format is available in the file *PTZ_Abstractions.py*
Eagle codes extends this file to train policies for different scenarios. The Eagle codes are available in the folder *Eagle_Codes*.

Running the  *PTZ_Abstractions.py* randomly modifies the Pan-Tilt-Zoom parameters of the camera along with capture the image from PTZ camera
for 100 steps.

```
python PTZ_Abstractions.py
```


## 2. Eagle: End-to-end Deep-RL for PTZ

We provide the source code to train and evaluate Eagle policies in the folder *Eagle_Codes*.
The sub-folders contains instructions to run the codes.


## 3. EagleSim Source Code
- The source code of EagleSim is [available](https://drive.google.com/drive/folders/1bSXGXncTOqWH2KirhMq8_iCwEEPqxJTC?usp=sharing) in the file *Cars_Trees_humans_Unreal_Source_Code.zip*
-  The compilation of the source code is done by following the steps listed [here](https://microsoft.github.io/AirSim/unreal_custenv/).
- The virtual world included is custom created with several actors, pawns, and blueprints.
- To avoid compilation, use the provided binaries.
- AirSim code is included as a plugin and is modified to enable human tracking.
