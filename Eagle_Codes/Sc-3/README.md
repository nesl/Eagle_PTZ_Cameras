# SC-3 Scenario: Variable backgrounds + Trees + Image Augmentations

Details to train for vehicle tracking scenario for Sc-3.
CustomPawnBlue_SUV1 is present in *settings_train.json*, which refers to the BlueSuv1 vehicle.
*settings_train.json* also enables the tree placements and variable backgrounds during training.
The abstractions of the scene controller are provided *settings_train.json*, which selectively enables scene complexity.
*ControlCar.py* provides the capability to control a vehicle on random trajectories.
*train.py* file provides PTZ abstractions of reward calculations and camera control in an open-AI gym format and uses stable-baselines3 for training.

## Training steps
1. Start the virtual world by putting *settings_train.json* to the appropriate folder for settings of the AirSim simulator.  The folder location is generally */HomeFolder/Documents/AirSim* in Linux machines. Rename the file to *settings.json*, which the EagleSim will read. The *settings.json* is an extension of the AirSim settings file with more options to enable different training scenarios.

2. This training is done in parallel; particularly, we assume 6 parallel scenes are running so as to speed up the data collection. The parallel scenes are started on different ports by modifying the *"ApiServerPort"* in *settings.json*, and the binary of vehicle tracking is started from different terminals to start parallel scenes. We use the following ports [41451, 41452, 41453, 41454, 41455, 41456].

3. Enable random trajectories of the vehicle using the *ControlCar.py* file. In *ControlCar.py*, random trajectories must be enabled for each parallel scene. This is done by changing the *global_port = 41451* to the 6 different ports used in step 2. So the following command is run from 6  terminals.
```
python ControlCar.py
```

3. Start the training using the *train.py* file. The training is done using parallel scenes started in step 1. The training files have open-AI gym abstractions and delay control. We recommend training using the end-to-end delay between 25-30ms. *train.py* has comments for clarification of different variables used. *tmp_path* is the location where training checkpoints will be saved. The policy is updated every 4096 steps from each scene (a total of 4096*6 steps per iteration). We recommend training for >2000 iterations which takes around three days with parallel data collection.
```
python train.py
```

4. During training, the PTZ camera view can be visualized using:
```
python ViewCamera.py
```

## Evaluation steps
1. Start the virtual world by putting *settings_evaluate.json* to the appropriate folder for settings of the AirSim simulator.  The folder location is generally */HomeFolder/Documents/AirSim* in Linux machines. Rename the file to *settings.json*, which the EagleSim will read.


2. Enable random trajectories of the vehicle using the *ControlCar.py* file, assuming the virtual world is running on 41451 port. You can also change *global_port* to desired values.
```
python ControlCar.py
```

3. Start the evaluation using the *evaluation.py* file. We recommend using the end-to-end delay between 25-30ms. A trained checkpoint is provided in the *Trained_sc1* folder. The file saves the performance data, including the steps completed (out of 2000), tests for 100 episodes, and displays data on the terminal for a visual view.

```
python evaluate.py
```

4. Evaluation of complex scenes is enabled by modifying the *settings.json*. Several options in the settings create more complex PTZ tracking scenes when enabled. Samples evaluation files [*settings_evaluate1.json*, *settings_evaluate2.json*] are provided in this folder.  
*settings_evaluate1.json* can be used to evaluate *Variable backgrounds+Trees*.
*settings_evaluate2.json* can be used to evaluate *Variable backgrounds+Trees+Humans*.
