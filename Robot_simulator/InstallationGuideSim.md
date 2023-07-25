# How to set up your Simulator Environment
This readme provides a guide on using the simulator environment.

Please see the [Installation Guide for physical robot](../Week01-02/InstallationGuidePhysical.md) on how to [install Linux](../Week01-02/InstallationGuidePhysical.md#Installing-Linux) and [set up the environment](../Week01-02/InstallationGuidePhysical.md#install-the-environment-from-scratch-in-an-empty-ubuntu-18)


# Launch the Simulator Environment

From within the Ubuntu environment (VM/native/WSL2, whichever one you use), open a terminal and type
```
source ~/catkin_ws/devel/setup.bash
roslaunch penguinpi_gazebo ECE4078.launch
```
You should see a base Gazebo world open with PenguinPi inside an empty map
![Empty World](EmptyWorld.png?raw=true "Empty World")

This terminal window will need to stay open while you are working with the simulated robot. **Keep an eye on the messages in this terminal window**, if there are only normal messages (text color in white) and warnings (text color in yellow) then all is good. If you see error messages (text color in red) then you will need to double check if all required packages are installed (see [set up the environment](../Week01-02/InstallationGuidePhysical.md#install-the-environment-from-scratch-in-an-empty-ubuntu-18)). For example, without the ```flask``` python package you won't be able to establish a connection to control the virtual robot spawn inside Gazebo, and you will see an error message about the penguinpi server failing to start.

**Open a new terminal** and spawn objects in the map by typing the following commands (right click on scene_manager.py and go to "Properties -> Permissions" to double check that the "Execute" box is ticked. This file is under the ~/catkin_ws/src/penguinpi_gazebo/ folder)
```
source ~/catkin_ws/devel/setup.bash
rosrun penguinpi_gazebo scene_manager.py -l map1.txt
```
You should see the empty map now populated with targets and markers.
Play around with the ```rosrun penguinpi_gazebo scene_manager.py``` command using the following variations:
- Run without ```-l map1.txt``` tag to spawn objects at random locations
- Run with ```-d``` tag to remove the spawned objects
- Run with ```-s NAME.txt``` to save a new map

You can change the number of targets and markers in the map by changing obj_class_dict at Line 23 of scene_manager.py

You can include more virtual objects under the ~/catkin_ws/src/penguinpi_gazebo/models/ folder, see [this tutorial](https://classic.gazebosim.org/tutorials?tut=import_mesh&cat=build_robot) for more information. 

![Simulator Map](SimulatorMap.png?raw=true "Simulator Map")

