# Approach-3: Relative_location + Control

Approach-3 is a multi-stage pipeline using relative locations as input to a controller.
Here, we provide the training code (*train.py*). The evaluation (*evaluate.py*) is on the BlueSuv1 vehicle (CustomPawnBlue_SUV1 is present in *settings.json*).

*settings.json* also enables the tree placements, variable backgrounds, and human characters during evaluation.
The abstractions of the scene controller are provided in *settings.json*, which selectively enables scene complexity.
*ControlCar.py* provides the capability to control a vehicle on random trajectories.
Training of the Approach-3 is done using relative locations in a supervized manner from the images data from the EagleSim. *train.py* file provides the training code. A trained checkpoint of model is provided in this folder.

## Training steps
1. Download the [dataset](https://drive.google.com/drive/folders/1Lxh0QjHha4cKoYzTyZ6ioQgoTzrCQfle?usp=sharing) to train the model. We use the dataset (images_240_50k.npz and labels_240_50k) of 50k images to train the model.

2. This training is done in a supervized manner; particularly, a model is trained to predict the relative locations of vehicle in the image.
In *train.py*  update the variables to point to the path of the downloaded dataset (*file_name_img* and *file_name_labels*).
Start the training using the following command.
```
python train.py
```


## Evaluation steps
1. Start the virtual world by putting *settings.json* to the appropriate folder for settings of the AirSim simulator.  The folder location is generally */HomeFolder/Documents/AirSim* in Linux machines. The *settings.json* is an extension of the AirSim settings file with more options to enable different objects in PTZ tracking scenarios.

2. The scenes can be started on different ports by modifying the *"ApiServerPort"* in *settings.json*, and the binary of vehicle tracking is started from different terminals to start parallel scenes. The code uses the port [41451] to run.

3. Enable random trajectories of the vehicle using the *ControlCar.py* file. In *ControlCar.py*, random trajectories must be enabled for each scene that is tested. This is done by changing the *global_port = 41451* to the desired port used in step 2. The following command is run from a terminal.
```
python ControlCar.py
```

4. Start the evaluation using the *evaluate.py* file. Evaluation saves the performance data, including the steps completed (out of 2000), tests for 100 episodes, and displays data on the terminal for a visual view.
```
python evaluate.py
```

4. During an evaluation, the PTZ camera view can be visualized using:
```
python ViewCamera.py
```

5. Evaluation of complex scenes is enabled by modifying the *settings.json*. Several options in the settings create more complex PTZ tracking scenes when enabled.
*settings_evaluate.json* can be used to evaluate *Fixed background*.
*settings.json* can be used to evaluate *Variable backgrounds+Trees+Humans*.
