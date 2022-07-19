# Dynamic Tasking: Enable tasking of policies at runtime using contextual input

Details to train for vehicle/human tracking scenario for DT (dynamic tasking).
The training scenes are created using *CustomPawnBlue_SUV1* for vehicle tracking binary and 4 random human characters
are created by human tracking binary. The setting file for vehicle tracking binary is *settings_vehicle.json*, and for human tracking binary is *settings_humans.json*.

*settings.json* also enables the tree placements and background variations during training.
The abstractions of the scene controller are provided *settings.json*, which selectively enables scene complexity.
*ControlCar.py* provides the capability to control a vehicle on random trajectories.
Human characters are coded to move on random trajectories in the human tracking virtual world.
*train.py* file provides PTZ abstractions of reward calculations and camera control in an open-AI gym format and uses stable-baselines3 for training.

The major change is the contextual input concept in *train.py*, where for vehicle tracking virtual world data, we use a different contextual input than the human tracking virtual world data. The reward reshaping is done in the *train.py* file.

## Training steps
1. Start the vehicle virtual world by putting *settings_vehicle.json* to the appropriate folder for settings of the AirSim simulator.  The folder location is generally */HomeFolder/Documents/AirSim* in Linux machines. Rename the file to *settings.json*, which the EagleSim will read. The *settings.json* is an extension of the AirSim settings file with more options to enable different training scenarios.
Start the human tracking virtual world by using *settings_humans.json*, and follow the similar steps as earlier.

2. This training is done in parallel; particularly, we assume 6 parallel scenes are running so as to speed up the data collection. The parallel scenes are started on different ports by modifying the *"ApiServerPort"* in *settings.json*, and the binary of vehicle tracking is started from different terminals to start parallel scenes. We use the following ports [41451, 41452, 41453, 41454, 41455, 41456]. We start first three scenes (ports: [41451, 41452, 41453]) for vehicle tracking and the rest three scenes for human tracking.


3. Enable random trajectories of the vehicle using the *ControlCar.py* file. In *ControlCar.py*, random trajectories must be enabled for each parallel scene. This is done by changing the *global_port = 41451* to the 3 different ports used in step 2. So the following command is run from 3  terminals. Human characters are by default coded to move on random trajectories in the human tracking virtual world.
```
python ControlCar.py
```

3. Start the training using the *train.py* file. The training is done using parallel scenes started in step 1. The training files have open-AI gym abstractions and delay control. We recommend training using the end-to-end delay between 25-30ms. *train.py* has comments for clarification of different variables used. *tmp_path* is the location where training checkpoints will be saved. The policy is updated every 4096 steps from each scene (a total of 4096*6 steps per iteration). We recommend training for 2000 iterations or more for tasking which takes around three days with parallel data collection.
```
python train.py
```

4. During training, the PTZ camera view can be visualized using:
```
python ViewCamera.py
```

## Evaluation steps
1. Start the vehicle tracking virtual world by using *settings_vehicle.json* and human tracking virtual world by using *settings_humans.json* as done during training. However, here only one scene from each virtual world is used for evaluations.  The folder location is generally */HomeFolder/Documents/AirSim* in Linux machines. Rename the file to *settings.json*, which the EagleSim will read.


2. Enable random trajectories of the vehicle using the *ControlCar.py* file, assuming the virtual world is running on 41451 port. You can also change *global_port* to desired values. Human characters are by default coded to move on random trajectories in the human tracking virtual world.
```
python ControlCar.py
```

3. Start the evaluation using the *evaluation.py* file. We recommend using the end-to-end delay between 25-30ms. A trained checkpoint is provided in the *Trained_sc1* folder. The file saves the performance data, including the steps completed (out of 2000), tests for 100 episodes, and displays data on the terminal for a visual view. To track humans or cars, change the contextual input in the *evaluation.py*, as mentioned in the comments in the starting.

```
python evaluate.py
```

4. Evaluation of complex scenes is enabled by modifying the *settings.json*. Several options in the settings create more complex PTZ tracking scenes when enabled.
