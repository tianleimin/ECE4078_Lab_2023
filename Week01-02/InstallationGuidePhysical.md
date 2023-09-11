# How to set up your Robot
This readme provides a step-by-step guide on setting up your environment and connecting to the PenguinPi robot.

**For setting up the environment, we recommend Ubuntu 18 in this guide, as it works for both simulator and physical robot. However, as all the markings are done on the physical robot, the simulator environment is NOT required. You only need to [install the required python packages](#Installing-Python-env-to-work-with-a-physical-robot) if you don't plan to use the simulator environment. The python env can be installed on your home system directly, whether it's Ubuntu 18 or not.** 

For information on using the simulator environment please see [the guide on the simulator environment](../Robot_simulator/InstallationGuideSim.md#Launch-the-Simulator-Environment).

- [Installing Python env to work with a physical robot](#Installing-Python-env-to-work-with-a-physical-robot)
- [Installing Linux](#Installing-Linux)
    - [Option 1: using the pre-built VM](#option-1-using-the-pre-built-virtual-machine-image)
    - [Option 2: using WSL2](#option-2-installing-wsl2-on-windows)
    - [Option 3: dual boot your computer](#option-3-dual-boot-your-computer)
- [Install the environment from scratch (Only for Option 2 & 3)](#install-the-environment-from-scratch-in-an-empty-ubuntu-18)
- [Running on the Physical Robot](#Running-on-the-Physical-Robot)
    - [Inside the robot kit](#inside-the-kit)
    - [Powering on and connecting to the robot](#powering-on-and-connecting-to-the-robot)
    - [Running codes on the robot](#running-codes-on-the-robot)
    - [Disconnecting and shutting down the robot](#disconnecting-and-shutting-down-the-robot)
- [Troubleshooting](#Troubleshooting)

---
# Installing Python env to work with a physical robot
If you only plan to work with the physical robot, then you only need to install required python packages on your home system and there is no need to install the Linux environment.

1. Install a [Python](https://www.python.org/downloads/) (use version ≥ 3.6)

2. Navigate to a directory where you want to install the robot-related stuff and open it in a terminal

3. Create a Python virtual environment named "PenguinPi" (this is optional, setting up a venv helps to ensure the robot-related packages won't interfere with your other Python stuff): ```py -3 -m venv PenguinPi```

4. Activate your venv: ```PenguinPi\Scripts\activate```

5. Install pip in the venv: ```python -m pip install -U pip```

6. Install wheel in the venv: ```python -m pip install wheel```

7. Install python packages needed for M1 in the venv: ```python -m pip install flask gevent pyyaml numpy requests opencv-python pynput pygame```

Note: if you follow these steps to set up a python venv, every time before running robot-related codes you'll need to activate the venv. Also when running a script, instead of ```python3``` you will use ```python```, e.g., ```python operate.py --ip 192.168.50.1 --port 8080```.

8. To deactivate the env after you are done working inside it type ```deactivate```

**If you don't plan to use the simulator, then you may skip ahead to [Running on the Physical Robot](#Running-on-the-Physical-Robot)**

---
# Installing Linux 
We provide three options here for you. Option 1 is the easiest solution but it may not give you the best performance computation-wise. The other two options would likely give you boost in computational performance, but may require a bit more work to get things set up. We strongly recommend you to stick with Option 1 if you are not comfortable with setting up your own ROS environment. **You only need to install the Linux environment if you plan to use the simulator**

Note that for people with Mac (especially with Apple M1/M2 chips), please work with the demonstrators to see what works best, as some of these instructions may not work for you.

## Option 1: Using the pre-built virtual machine image 

Install [Oracle VM Virtualbox](https://www.virtualbox.org/) (make sure your [BIOS setting allows virtualization](https://forums.virtualbox.org/viewtopic.php?f=1&t=62339))

Download [the ready-to-use VM image](https://drive.google.com/file/d/1_MRbKHOA1Lo_qFBaASKvvaQ36HuJRIPI/view?usp=sharing) and [import](https://docs.oracle.com/en/virtualization/virtualbox/6.0/user/ovf.html#ovf-import-appliance) it to your VirtualBox (username: ```ece4078```  password: ```2021Lab```). 

You can change the amount of resources (e.g., RAM, processing cores) assigned to your VM in VirtualBox settings after importing the VM image depending on the hardware available to you.

## Option 2: Installing WSL2 on Windows
The new Windows Subsystem for Linux (WSL2) has nice supports for GUI app (eg. Gazebo), and it may run faster on your laptop/PC compared to running in a virtual machine. However, the nice GUI support for WSL2 only works well on Windows 11 (and certain Windows 10 build), so this option may not be viable for you if you don't have Windows 11. 

Please follow these [instructions](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#2-install-wsl) to set up your WSL2. Note that you should install **Ubuntu 18.04**, and you can skip the step to install octave.

To access the files in your Win11 from the Ubuntu subsystem, in your ubuntu terminal you can use ```cd /mnt/```. For example, if you have a folder named ```test_folder``` in the C:\ drive of Win11, you can navigate to that directory using ```cd /mnt/c/test_folder```. To transfer files between Win11 and Ubuntu, you can use the ```cp``` command (more info [here](https://manpages.ubuntu.com/manpages/trusty/man1/cp.1.html)). For example, if you have a file at C:/test.txt in Win11, you can copy it to your Ubuntu using ```cp /mnt/c/test.txt your_destination_dir```.

Otherwise, you may try [this](https://devblogs.microsoft.com/commandline/access-linux-filesystems-in-windows-and-wsl-2/) if you prefer using explorer to drag and drop. You may place your files/folders in ```/home/your_user_name/```.

## Option 3: Dual boot your computer 
CAUTION! Please only proceed with this option if you know what you are doing, and do this at your own risk.

Here is a [tutorial](https://www.youtube.com/watch?v=CWQMYN12QD0&ab_channel=TechnoTim) you can follow to set up dual boot on your computer. 

# Install the environment from scratch in an empty Ubuntu 18

***Note: ROS-Melodic requires Ubuntu 18, and you only need to do this if you have chosen either Option 2 or 3***

This instruction allows you to install the environment that supports development with both the physical and simulated robot. If you are only working with the physical robot, you only need to install the python packages. We provide the simulation environment as an additional resource since it can be useful for development and debugging when you don't have access to a physical robot or an arena, or when you want to have precise control over an object's coordinates and behaviours for evaluating a function.

## Python packages
install python packages
```
sudo apt update
sudo apt install python3-pip curl python-flask python-gevent
python3 -m pip install --upgrade pip 
python3 -m pip install flask gevent pyyaml numpy requests opencv-python pynput pygame
```

## ROS + Gazebo for simulation
install Gazebo 11 (after the installation is done you should be able to open Gazebo by typing ```gazebo``` in your terminal)
```
curl -sSL http://get.gazebosim.org | sh
```

install ROS Melodic
```
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
sudo apt install ros-melodic-desktop
```

install ROS Melodic packages for Gazebo 11
```
sudo apt install ros-melodic-gazebo11-plugins
sudo apt install ros-melodic-gazebo11-ros-pkgs 
sudo apt install ros-melodic-gazebo11-ros-control
sudo apt install python-catkin-tools python3-dev python3-catkin-pkg-modules python3-rospkg-modules python3-numpy python3-empy
```

setup local catkin workspace
```
source /opt/ros/melodic/setup.bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin_make
```

download the PenguinPi modules
```
cd ~/catkin_ws/src
sudo apt install git
sudo apt update
git clone https://bitbucket.org/cirrusrobotics/penguinpi_description.git
git clone https://bitbucket.org/cirrusrobotics/penguinpi_gazebo.git
git clone -b melodic https://github.com/ros-perception/vision_opencv.git
cd ~/catkin_ws
catkin_make
```

open ```~/catkin_ws/penguinpi_gazebo/scripts/server``` in a text editor, find [line 399-404](https://bitbucket.org/cirrusrobotics/penguinpi_gazebo/src/5760f245291dc639e146fd4adc3f5328333c2ef7/scripts/server#lines-399) and replace the codes with the following:
```
serverport = 40000
http_server = WSGIServer(('', serverport), app)
http_server.serve_forever()
```

**[EDIT]** Please replace the ```src``` folder in your local ```catkin_ws``` directory with [this src folder provided](https://drive.google.com/file/d/1VSmSmg7iuF-tQzWhlnmtwjsNhzxlPqU8/view?usp=sharing), which contains the required models, launch / world files, and the ```scene_manager.py``` script, then run ```catkin_make``` again under ```catkin_ws``` with the downloaded ```src``` folder).

---
# Running on the Physical Robot

## Inside the kit

The PenguinPi robot kit includes:
* one PenguinPi robot: details see fig below
* one Wifi Dongle: to be plugged in the robot's usb port for generating wifi hotspot for a computer to connect to the robot
* one USB cable: an extender of any usb ports on the robot for dev and debugging
* one charging cable: when fully charged, the light on the charger will change from red to green

The robot has an ID number printed on the carrying case and on the robot for logging ownership and repairs. Please don't mismatch the robot and its case.

![PenguinPi Robot Kit](PenguinPiKitAnno.png?raw=true "PenguinPi Robot kit")

Here is an overview of the robot's parts. **The camera is the front side of the robot** (the triangle part of the frame is the tail).

![PenguinPi Robot](PenguinPi.png?raw=true "PenguinPi Robot")

## Powering on and connecting to the robot
First, switch the robot on with the side switch

Wait for it to boot - this takes about 1 min. When IP addresses appears on the OLED screen, e.g., ```192.168.50.1```,  the booting is finished.

Connect your PC's wifi to the PenguinPi's hotspot penguinpi:xx:xx:xx (look for it in your wifi list). The hotspot password is ```PenguinPi```. You can identify which wifi belongs to your robot by the WMAC shown on the robot's screen (the last 6 digits/letters shown would match the hotspot's last 6 digits/letters).

You can test your connection by opening a web browser and entering the address ```192.168.50.1:8080``` ("192.168.50.1" is your robot's IP, "8080" is the port). On this webpage you will be able to test the motors and camera, as well as see a range of diagnostic information relating to your robot. **Please let us know if your robot's motors and/or camera aren't working so we can arrange for replace and repair.**

![Web browser view of the physical robot](WebRobot.png?raw=true "Web browser view of the physical robot")

You can also connect to the robot using ssh (the ssh password is ```PenguinPi```), replace the IP address if needed.

```
ssh -X pi@192.168.50.1
```

You can upload files to the robot using the ```scp``` command:
```
scp -r ./LOCALDIR pi@192.168.50.1:~/REMOTEDIR
```

## Running codes on the robot
To run your codes on the robot, you need to find the IP address of your robot. This is shown on its OLED screen. When you run ```operate.py``` you will need to include the "--ip" and "--port" flags to your python running command, for example ```python3 operate.py --ip 192.168.50.1 --port 8080```. The python script will then be executed on the physical robot, instead of on the simulated robot. 

After your environment is set up, in addition to working with the physical robot, you can also [launch the simulator environment](../Robot_simulator/InstallationGuideSim.md#Launch-the-Simulator-Environment) to work with a simulated robot.

You can also connect an external screen, keyboard, and mouse to the robot, then switch it on and use it as a Raspberry Pi. Inside the Raspberry Pi interface, you can install python packages onboard of the robot by running pip in the terminal, e.g., ```python3 -m pip install pynput```. You can also install packages inside the ssh session if your PenguinPi has internet connection (you can set the internet connection up in the Raspberry Pi interface).

## Disconnecting and shutting down the robot
When you are done with the robot, inside the ssh session run ```sudo halt & logout``` to shut down the robot safely. Once the screen says 'TURNRaspberryPi OFF SAFELY' you can toggle the power switch. Avoid toggling the switch directly to "hard" shut down.

---
# Troubleshooting

- Please be gentle with the robot as they can be quite fragile
- Some of the robots may come with one or both of the wheels soldered differently than the simulator (you can check this using the 'Test motors' function on the web browser view, the Left/Right buttons should make the left/right wheel each spin **forward** briefly to be consistent with the sim). Modify the ```set_velocity``` function in [pibot.py](util/pibot.py#L21) and the ```control``` function in [operate.py](operate.py#L58) to fit how your robot is assembled. Most commonly seen type is the **right wheel rotates backwards** when running motor test, which has already been addressed in the current codes. Please let us know if one of both of your wheels have directions different from the sim so we can keep a log on this.
- Some of the robots may have a faulty camera (you can check this via the camera stream window on the web browser view). Please let us know if you have a robot with a faulty camera so we can arrange for replacement and repair.
- Some of the robots may have a faulty battery (if the charger LED blinks red when you are charging the robot and plug-unplug several times doesn't fix this). Please let us know if you have a robot with a faulty battery so we can arrange for replacement and repair.
- Virtual box not importing the image properly (not able to import and open the image at all): if the error is related to E_INVALIDARG, check if your virtualbox has been installed in C:\ (instead of D:\ or any other drive).
- Don't tick the 3D acceleration option in Virtualbox as it might cause the VM to black-screen or crash with an out-of-memory error.
- Razer graphic cards might have issues with the VM/gazebo/RViz (crashes without apparent error messages).
- If running the VM on a Mac, and encounter a "Kernel Driver Not Installed (rc=-1908 error)" follow [this guide](https://www.howtogeek.com/658047/how-to-fix-virtualboxs-%E2%80%9Ckernel-driver-not-installed-rc-1908-error/) to fix the issue.
- If installing from scratch and the command to install Gazebo: "curl -sSL http://get.gazebosim.org | sh" does nothing, follow the instructions under "Alternative installation: step-by-step" in [this guide](http://gazebosim.org/tutorials?tut=install_ubuntu)
- Slow VM on Mac: this might be a resolution issue, see [here](https://www.reddit.com/r/virtualbox/comments/houi9k/how_to_fix_virtualbox_61_running_slow_on_mac/) for fixes. You can also try [enabling 3D acceleration](https://superuser.com/questions/172989/virtualbox-running-ubuntu-is-slow-mac-os-x-host) in VistualBox

---
# Acknowledgement
- Part of the lab sessions are inspired by the [Robotic Vision Summer School](https://www.rvss.org.au/)
- Robot dev in collaboration with [QUT Centre for Robotics](https://github.com/qcr/PenguinPi-robot)
