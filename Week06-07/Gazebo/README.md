# Milestone 3: Object Recognition and Localisation
- [Introduction](#introduction)
- [Useful resources](#an-index-of-provided-resources)
- [Data collection (Week 6)](#step-1-data-collection)


## Introduction
For milestone 3 you are required to detect and localise a list of target objects by training a target detector. In Week 6, you need to collect the training data and train your target detector with the helper scripts. In Week 7, you need to complete the codes to estimate the pose of the detected targets. 

The following is only a guideline to assist you to train a basic target detector that can provide sufficient performance for you to complete this milestone. However, you may want to explore using other state of the art object detection models such as [YOLOv5](https://github.com/ultralytics/yolov5). This guideline provides helper scripts for you to automate the data generation for the simulation. However, you will need to collect your own data set for detecting the fruits with the physical robot. We encourage you to write your own scripts to automate the training data generation for the real life data set. We have provided some suggestions on how to do so in instructions below. You may combine your synthetic data with the real data together for training a single model, or you can train two separate models for the simulation and the physical robot.

## An index of provided resources
- [operate.py](operate.py) is the central control program that combines keyboard teleoperation (M1), SLAM (M2), and object recognitions (M3). It requires the [utility scripts](../Week01-02/util), the [GUI pics](../Week01-02/pics) from M1, and the [calibration parameters](../Week03-05/calibration/param/) and [SLAM scripts](../Week03-05/slam/) of M2. In addition, it calles for the [detector model](#Training-your-neural-network) that you will develop in M3.
- [ECE4078_brick.world](ECE4078_brick.world) and [ECE4078_brick.launch](ECE4078_brick.launch) are used for launching a new testing arena with wall and floor textures added.
- [models.zip](https://drive.google.com/file/d/1-RMIWwW1THG4xKS_XeNMGs168aqACDL0/view?usp=sharing) contains the new fruit models for the simulation
- [FruitMap.txt](FruitMap.txt) is an example map to spawn the fruit models for testing
- [data_collector.zip](https://drive.google.com/file/d/1zxjZpJVTPvO_RGJGkwHJBu0dJRlktOqe/view?usp=sharing) contains scripts required for generating the [synthetic dataset](#Data-collection) used for training your target detector.

## Step 1: Data collection
### a) Copying the additional files
1. Download [ECE4078_brick.world](ECE4078_brick.world) and [ECE4078_brick.launch](ECE4078_brick.launch) to your ```worlds``` and ```launch``` folder inside ```catkin_ws/src/penguinpi_gazebo```

2. Unzip the [models.zip](https://drive.google.com/file/d/1-RMIWwW1THG4xKS_XeNMGs168aqACDL0/view?usp=sharing) folder to ```catkin_ws/src/penguinpi_gazebo/models```. This models folder contains new fruit models for the simulation world: orange, pear and strawberry  

3. Copy [scene_manager.py](scene_manager.py) to the ```catkin_ws/src/penguinpi_gazebo``` folder. Line 24 to 26 have been updated

4. Unzip [data_collector.zip](https://drive.google.com/file/d/1zxjZpJVTPvO_RGJGkwHJBu0dJRlktOqe/view?usp=sharing) to ```catkin_ws/src/```

### b) Changing environment textures
Now open a terminal and run: 
```
catkin_make or catkin build
source ~/catkin_ws/devel/setup.bash
``` 
The data_collector package will now be recognised by ROS. Now run

```
roslaunch penguinpi_gazebo ECE4078_brick.launch
``` 
You should see an areana with brick walls and wooden floors. You can then spawn objects inside this new environment by opening a new terminal and run 
```
source ~/catkin_ws/devel/setup.bash
rosrun penguinpi_gazebo scene_manager.py -l /path_to_file_dir/FruitMap.txt
```
Try changing the wall and floor materials during development to test the robustness of your detector. To do so, open [ECE4078_brick.world](ECE4078_brick.world) in a text editor, search and replace "Gazebo/Bricks" with other pre-defined materials listed [here](http://wiki.ros.org/simulator_gazebo/Tutorials/ListOfMaterials) to change the wall material (4 occurrences in total), and search and replace "Gazebo/Wood" with other pre-defined materials to change the floor material (1 occurrence in total).

![Brick Walls](Screenshots/BrickWallWorld.png?raw=true "Brick Walls")

### c) Generating synthetic data
As training a neural network requires a lot of data, but manual data collection is time consuming, a ROS-based [data collector](https://drive.google.com/file/d/1zxjZpJVTPvO_RGJGkwHJBu0dJRlktOqe/view?usp=sharing) is provide for you to generate synthetic data for training. 

Open a terminal and install required packages for **python 2** used by the ROS data collector:
```
sudo apt install python-pip
python -m pip install tqdm h5py
```
In catkin_ws/src/data_collector/data_collector.py, replace the camera matrix at Line 33 with your [camera parameters](../Week03-05/calibration/param/intrinsic.txt) computed from the camera calibration process in M2.

**Close Gazebo if it is running.** Then open the Gazebo photo studio (an empty world with lighting set-up) by running the following command in a terminal 
```
source ~/catkin_ws/devel/setup.bash
roslaunch data_collector photo_studio.launch gui:=true
```

Now open a new terminal and generate the synthetic dataset by typing the commands below (note that we are using "python" instead of "python3")
```
source ~/catkin_ws/devel/setup.bash
roscd data_collector
python data_collector.py --dataset_name Synth --sample_per_class 1000
```
This creates a folder in "catkin_ws/src/data_collector/dataset/" called "Synth", in which there is a folder for each target class containing 1000 synthetic training images generated with grey background in the images folder (if you check the Gazebo window while the data is being generated, you can see the models appearing at random locations for taking each of these images). The "images" folder saves the original image, while the "labels" folder saves the segmentation labels (silhouette of the target model) with small variation in the black level for training your neural network. The black level variation is amplified in "labels_readable" so that you can visualize what the labels look like if you want.

![pear model with grey background](Screenshots/pear_grey.jpg?raw=true "Pear model with grey background")

Now we need to replace the grey background with random background images to increase the detector's robustness. Open a terminal and run the following commands (note that we are using "python", not "python3")
```
cd ~/catkin_ws/src/data_collector/
python randomise_background.py --dataset_name Synth
``` 
You should now see the images in "catkin_ws/src/data_collector/dataset/Synth" with random background pictures added to them. The background pictures are stored in "catkin_ws/src/data_collector/textures/random" and you can remove part of it or add your own collection.

![Pear model with random background](Screenshots/pear_rand.jpg?raw=true "Pear model with random background")

In the same terminal, after the background shuffling is done, run the dataset split script (note that we are using "python", not "python3"):
```
python split_dataset.py --sim_dataset Synth --training_ratio 0.8
``` 
This will separate the synthetic dataset randomly into a training set containing 80% of all images and an evaluation set containing 20% of all images, specified by "train.hdf5" and "eval.hdf5" generated under "catkin_ws/src/data_collector/dataset/", which will be used for training your neural network.
