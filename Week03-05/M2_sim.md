# M2: SLAM in Simulation
- [Sep up the sim env](#Sep-up-the-sim-env)
    - [Additional python packages](#Additional-python-packages)
    - [Additional sim models](#Additional-sim-models)
- [Activities](#Activities)
    - [Calibration (week 3)](#Calibration-week-3)
    - [SLAM (week 4-5)](#SLAM-week-4-5)

---
## Sep up the sim env
### Additional python packages
You'll need to install required packages by typing the following commands in the terminal:
```
python3 -m pip install machinevision-toolbox-python spatialmath-python==0.8.9 opencv-contrib-python==4.1.2.30 matplotlib
cd ~/catkin_ws/
source ~/catkin_ws/devel/setup.bash
catkin_make
```

To confirm the required packages are installed, in terminal type ```python3```, which opens python 3 in command line, then type:
```
import cv2
from cv2 import aruco
from machinevisiontoolbox import Image, CentralCamera
```
If there is no error then everything is successfully installed and you can exit the command line python 3 by typing ```exit()```


### Additional sim models

Copy the [calibration rig model](gazebo_calib/rig) to your ```~/catkin_ws/src/penguinpi_gazebo/models/```

Copy the [calibration rig launch file](gazebo_calib/calibration.launch) to your ```~/catkin_ws/src/penguinpi_gazebo/launch/```

Copy the [calibration rig world file](gazebo_calib/calibration.world) to your ```~/catkin_ws/src/penguinpi_gazebo/worlds/```

---
## Activities
### Calibration (week 3)

#### Step 1) Preparing Working directory
Prior to working on this week's materials, please make sure you do the following:
- Copy over the [util folder](../Week01-02/util) and the [GUI pics](../Week01-02/pics) from the M1 lab into your working directory
- Replace the [keyboard control section](operate.py#L194) in operate.py with the code you developed for M1

#### Step 2) Wheel Calibration (Simulation)
For wheel calibration inside the simulator, first launch the world by typing the following commands in your terminal (you don't need to generate any objects in the world for wheel calibration):
```
source ~/catkin_ws/devel/setup.bash
roslaunch penguinpi_gazebo ECE4078.launch
```

Then in another terminal, run the [wheel calibration script](calibration/wheel_calibration.py) using the command ```python3 wheel_calibration.py```. This script will set the robot driving forward, and then spinning at various velocities, for durations that you specify, in order to compute the scale and baseline parameters for wheel calibration.

On the Gazebo ground plane, each square grid is 1m^2. You can use that to measure if the robot has moved 1m, or rotated for 360deg. The script will prompt you to specify how long the robot should drive, or spin, and then confirm whether it has travelled or rotated the required amount.

It may be helpful to reset the robot's pose after each run. To do this, click on the robot, or select it from the list of models in the left panel. Expand the "pose" section in the left panel. To reset the robot's pose, and manually reset the values in 'x', 'y', and 'yaw' to 0. (Note: after you change its pose, the robot may not immediately show in Gazebo). Avoid using the translation (on z axis) /rotation mode on the robot as this will destroy its balance and take a long time to recover. 

If, after the robot has moved, the values still appear to be 0, select the robot from the list of models again (take care to ensure you haven't selected any of the links instead), and the values should update.

Try to be as accurate as possible when determining how far the robot has travelled, or spun, in order to improve the accuracy of your calibration calculations.

![Changing robot's pose in Gazebo](screenshots/GazeboPose.png?raw=true "Changing robot's pose in Gazebo")


#### Step 3) Camera calibration (Simulation)
To set up the calibration world in the simulator, copy the [rig folder](gazebo_calib/rig) to your catkin_ws/src/pernguinpi_gazebo/models/, [calibration.launch](gazebo_calib/calibration.launch) to your catkin_ws/src/pernguinpi_gazebo/launch/, and [calibration.world](gazebo_calib/calibration.world) to your catkin_ws/src/pernguinpi_gazebo/worlds/

You can then launch the camera calibration world using the commands below (first time launching the calibration world may take a little while):
```
source ~/catkin_ws/devel/setup.bash
roslaunch penguinpi_gazebo calibration.launch
```

![The calibration world in Gazebo](screenshots/GazeboCameraCalib.png?raw=true "The calibration world in Gazebo")

Take a photo of the calibration rig using [calib_pic.py](calibration/calib_pic.py) (located in the calibration folder). Run the script using the command ```python3 calib_pic.py```, and take a calibration photo by pressing ENTER while running the script.

You may also drive the robot around with arrow keys to a preferred location where all 8 calibration points are clearly visible before taking the photo. You will need to replace the [keyboard control section](calibration/calib_pic.py#L31) of this file with your codes from M1 to enable the driving functions. The calibration photo is saved as [calib_0.png](calibration/calib_0.png).

You will now need to calibrate the camera parameters, using using [camera_calibration.py](calibration/camera_calibration.py). This will require you to select 8 key points from the calibration photo you just took.

Run the script using the command ```python3 camera_calibration.py```. This opens the calibration photo you just took. Selecting the 8 key points in the calibration photo following the ordering shown in [calibration-fixture.png](diy_prints/calibration-fixture.png) by left clicking on each point (right click to cancel a selected point). Once all 8 points are selected, close the figure window to compute the camera matrix. This will update the [intrinsic parameters](calibration/param/intrinsic.txt). Note: keep the [distortion coefficients](calibration/param/distCoeffs.txt) to all 0s.

---
### SLAM (week 4-5)
Similar to working with the physical robot, you can develop and evaluate your SLAM in the simulator. 

You can generate and save new maps using ```rosrun penguinpi_gazebo scene_manager.py -s TRUEMAP.txt```. 

Note: the ```-s``` setting will generate a new map. If you want to use the default map1.txt (same as the example map provided) for testing, please use ```rosrun penguinpi_gazebo scene_manager.py -l map1.txt``` (this can be used to load any pre-defined map).

