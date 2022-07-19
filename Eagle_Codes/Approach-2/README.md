# Approach-2: Bounding_box + Reinforcement Learning

Approach-2 is a multi-stage pipeline using bounding boxes as input for deep-RL controller.
Here, we provide the training code (*train.py*) and two evaluation implementations (*PerfectBB.py* and *YoloBB.py*) of approach-2, which are evaluated on the BlueSuv1 vehicle (CustomPawnBlue_SUV1 is present in *settings.json*).


*settings.json* also enables the tree placements, variable backgrounds, and human characters during evaluation.
The abstractions of the scene controller are provided in *settings.json*, which selectively enables scene complexity.
*ControlCar.py* provides the capability to control a vehicle on random trajectories.
Training of the deep-RL model is done using perfect bounding boxes from the engine. *train.py* file provides the training code.

The evaluation is done using two ways.
*PerfectBB.py* and *YoloBB.py* files provide abstractions of calculating bounding boxes and doing the PTZ camera control.
*PerfectBB.py* uses bounding boxes from the EagleSim simulator, which are perfect, and have no errors.
*YoloBB.py* uses bounding boxes from the yolo5s model calculated using the image inputs from the EagleSim.
A tuned checkpoint of Yolo5s and deep-RL models are provided in this folder.

## Training steps
1. Start the virtual world by putting *settings.json* to the appropriate folder for settings of the AirSim simulator.  The folder location is generally */HomeFolder/Documents/AirSim* in Linux machines. The *settings.json* is an extension of the AirSim settings file with more options to enable different training scenarios.

2. This training is done in parallel; particularly, we assume 6 parallel scenes are running so as to speed up the data collection. The parallel scenes are started on different ports by modifying the *"ApiServerPort"* in *settings.json*, and the binary of vehicle tracking is started from different terminals to start parallel scenes. We use the following ports [41451, 41452, 41453, 41454, 41455, 41456].

3. Enable random trajectories of the vehicle using the *ControlCar.py* file. In *ControlCar.py*, random trajectories must be enabled for each parallel scene. This is done by changing the *global_port = 41451* to the 6 different ports used in step 2. So the following command is run from 6  terminals.
```
python ControlCar.py
```

3. Start the training using the *train.py* file. The training is done using parallel scenes started in step 1. The training files have open-AI gym abstractions and delay control. We recommend training using the end-to-end delay between 25-30ms. *tmp_path* is the location where training checkpoints will be saved. The policy is updated every 4096 steps from each scene (a total of 4096*6 steps per iteration). We recommend training for >2000 iterations which takes around three days with parallel data collection.
```
python train.py
```

4. During training, the PTZ camera view can be visualized using:
```
python ViewCamera.py
```

## Evaluation steps
1. Start the virtual world by putting *settings.json* to the appropriate folder for settings of the AirSim simulator.  The folder location is generally */HomeFolder/Documents/AirSim* in Linux machines. The *settings.json* is an extension of the AirSim settings file with more options to enable different objects in PTZ tracking scenarios.

2. The scenes can be started on different ports by modifying the *"ApiServerPort"* in *settings.json*, and the binary of vehicle tracking is started from different terminals to start parallel scenes. The code uses the port [41451] to run.

3. Enable random trajectories of the vehicle using the *ControlCar.py* file. In *ControlCar.py*, random trajectories must be enabled for each scene that is tested. This is done by changing the *global_port = 41451* to the desired port used in step 2. The following command is run from a terminal.
```
python ControlCar.py
```

4. Start the evaluation using the *PerfectBB.py* or *YoloBB.py* files. Evaluation saves the performance data, including the steps completed (out of 2000), tests for 100 episodes, and displays data on the terminal for a visual view.
```
python PerfectBB.py
```

4. During an evaluation, the PTZ camera view can be visualized using:
```
python ViewCamera.py
```

5. Evaluation of complex scenes is enabled by modifying the *settings.json*. Several options in the settings create more complex PTZ tracking scenes when enabled.
*settings_evaluate.json* can be used to evaluate *Fixed background*.
*settings.json* can be used to evaluate *Variable backgrounds+Trees+Humans*.
