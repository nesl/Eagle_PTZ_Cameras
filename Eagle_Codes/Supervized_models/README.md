# Approach: Bounding_box + Kalman + Controller

In this, the bounding box is computed for vehicles using simple supervised models having an architecture similar to the Eagle policy network.
It is a classical multi-stage pipeline doing object detection and tracking followed by control of PTZ parameters.
Here, we provide code to train supervised models(*Train_Supervized_model*) files, which are evaluated on the BlueSuv1 vehicle (CustomPawnBlue_SUV1 is present in *settings_simple.json* and  *settings_complex.json*).

*settings_complex.json* also enables the tree placements, variable backgrounds, and human characters during evaluation.
The abstractions of the scene controller are provided in *settings_complex.json*, which selectively enables scene complexity.
*ControlCar.py* provides the capability to control a vehicle on random trajectories.
*evaluate_models.py* provide abstractions of calculating bounding boxes and doing the PTZ camera control.
*evaluate_models.py* uses bounding boxes from the supervised models.

The controller has a rule-based design similar to the PTZ controller designed by [authors](https://www.sciencedirect.com/science/article/pii/S0957417421005911). The threshold values of the PTZ controller are tuned using [Mango](https://github.com/ARM-software/mango) to find optimal parameters maintaining the highest reward [(*center_x*)X(*center_y*)X(object_size)X(num_of_steps)].
We used the Kalman implementation from *Sort.py*, which is taken from open-source [Sort implementation](https://github.com/abewley/sort/blob/master/sort.py).

*Sort2.py* is the tuned Kalman filter. We tune the Kalman filter using [Mango](https://github.com/ARM-software/mango) by using the previously tuned controller to find the optimal Kalman parameters that achieve the highest reward [(*center_x*)X(*center_y*)X(object_size)X(num_of_steps)].

## Training supervised models
The model training code is available in *Train_Supervized_model* files. The training [dataset](https://drive.google.com/drive/folders/1Lxh0QjHha4cKoYzTyZ6ioQgoTzrCQfle?usp=sharing) to train the models is available. There are six different training files using [50k, 100k, 150k] images to train on either 120 by 120 or 240 by 240 images.

```
python Train_Supervized_model_50k_120.py
```


## Evaluation steps
1. Start the virtual world by putting *settings_complex.json* to the appropriate folder for settings of the AirSim simulator. The folder location is generally */HomeFolder/Documents/AirSim* in Linux machines. The *settings_complex.json* is an extension of the AirSim settings file with more options to enable different objects in PTZ tracking scenarios.

2. The scenes can be started on different ports by modifying the *"ApiServerPort"* in *settings.json*, and the binary of vehicle tracking is started from different terminals to start parallel scenes. The code uses the port [41451] to run.

3. Enable random trajectories of the vehicle using the *ControlCar.py* file. In *ControlCar.py*, random trajectories must be enabled for each scene that is tested. This is done by changing the *global_port = 41451* to the desired port used in step 2. The following command is run from a terminal.
```
python ControlCar.py
```

4. Start the evaluation using the *evaluate_models.py*. Evaluation saves the performance data, including the steps completed (out of 2000), tests for 100 episodes, and displays data on the terminal for a visual view. The trained checkpoints are provided in the *data* folder.
```
python evaluate_models.py
```

4. During an evaluation, the PTZ camera view can be visualized using:
```
python ViewCamera.py
```

5. Evaluation of complex scenes is enabled by modifying the *settings.json*. Several options in the settings create more complex PTZ tracking scenes when enabled.
*settings_simple.json* can be used to evaluate *Fixed background*.
*settings_complex.json* can be used to evaluate *Variable backgrounds+Trees+Humans*.

6. The performance is visualized in the *Visualize.ipynb* notebook.
