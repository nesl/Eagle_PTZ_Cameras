# Approach-1: Bounding_box + Kalman + Controller

Approach-1 is a classical multi-stage pipeline doing object detection and tracking followed by control of PTZ parameters.
Here, we provide two reference implementations (*PerfectBB.py* and *YoloBB.py*) of approach-1, which are evaluated on the BlueSuv1 vehicle (CustomPawnBlue_SUV1 is present in *settings.json*).

*settings.json* also enables the tree placements, variable backgrounds, and human characters during evaluation.
The abstractions of the scene controller are provided in *settings.json*, which selectively enables scene complexity.
*ControlCar.py* provides the capability to control a vehicle on random trajectories.
*PerfectBB.py* and *YoloBB.py* files provide abstractions of calculating bounding boxes and doing the PTZ camera control.
*PerfectBB.py* uses bounding boxes from the EagleSim simulator, which are perfect, and have no errors.
*YoloBB.py* uses bounding boxes from the yolo5s model calculated using the image inputs from the EagleSim.

The controller has a rule-based design similar to the PTZ controller designed by [authors](https://www.sciencedirect.com/science/article/pii/S0957417421005911). The threshold values of the PTZ controller are tuned using [Mango](https://github.com/ARM-software/mango) to find optimal parameters maintaining the highest reward [(*center_x*)X(*center_y*)X(object_size)X(num_of_steps)].
The Mango tuner for the controller is provided in the file (*ControllerTuner.py*). When tuning the controller, we used the default Kalman implementation from *Sort.py*, which is taken from open-source [Sort implementation](https://github.com/abewley/sort/blob/master/sort.py).

*Sort2.py* is the tuned Kalman filter. We tune the Kalman filter using [Mango](https://github.com/ARM-software/mango) by using the previously tuned controller to find the optimal Kalman parameters that achieve the highest reward [(*center_x*)X(*center_y*)X(object_size)X(num_of_steps)]. The Mango tuner used for Kalman is provided in the file (*KalmanTuner.py*).


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
