# Milestone 1: Teleoperating the Robot with your Keyboard

We will be using the [PenguinPi robot](https://cirrusrobotics.com.au/products/penguinpi/) for our lab project.

This first milestone will allow you to become more familiar with the PenguinPi robot and how to program it.

## Objective 1: Setting up your Environment
1. The [Installation Guide for physical robot](InstallationGuidePhysical.md) provides detailed instructions on setting up your environment and connecting to the robot.
2. The [Installation Guide for simulator](../Robot_simulator/InstallationGuideSim.md) provides detailed instructions on using a simulator environment for development. Please note that the simulator environment is provided as an additional support but is not required, as all markings will be done on the physical robot only.
3. Within your dev environment (VM/native/WSL2, whichever one you use), clone this repo by typing ```git clone https://github.com/tianleimin/ECE4078_Lab_2023``` inside your terminal.
4. Navigate to this week's lab by typing the commands ``` cd ECE4078_Lab_2023/Week01-02/```

## Objective 2: Implement Keyboard Teleoperation

You will implement keyboard teleoperations by editing [line 142 - 155 of operate.py](operate.py#L142).
  - Hint: check out how the stop control is implemented at [line 158](operate.py#L158) and the wheel control function at [line 57-74](operate.py#L57).
  - You will also need [the pics folder for painting the GUI](pics/) and [the util folder for utility scripts](util/), already contained within the repo.
  - To control the physical robot, you will need to include flag for its IP address and port in your command, for example, ```python3 operate.py --ip 192.168.50.1 --port 8080``` (see [Installation Guide for physical robot](InstallationGuidePhysical.md)).
  - Below is what the teleoperation GUI (and loading menu) looks like:

![GUI Menu](Menu.png?raw=true "GUI Menu")
![Teleop GUI](Teleop.png?raw=true "Teleop GUI")

**Note:** In order to run ```operate.py```, your terminal needs to be in the directory where [operate.py](operate.py) is (eg. ```ECE4078_Lab_2023/Week01-02/```)

**You don't have to use the provided scripts.** Feel free to be creative and write your own scripts for teleoperating the robot with keyboard.

[EDIT] If you don't want to type the ```--ip``` and ```--port``` args every time, simply change the default arg values ([line 176-177 in operate.py](operate.py#L176)) to the robot's ip ```192.168.50.1``` and port ```8080```, then the next time you run operate.py with the physical robot you won't need to specify the ip and port any more, simply run ```python3 operate.py```.
---

## Marking Instructions 
You will need to demonstrate your teleoperation implementation on the physical robot during the Week 3 lab session **as individuals**. 
If you have finished the implementation this week, feel free to ask the demonstrators to mark you during this week's session. 

Please follow the [marking steps](#Marking-steps) and submit your implementation to Moodle before asking to be marked.

- M1 live demo tasks by completing [operate.py](operate.py):
  - Drive forward +25pt
  - Drive backward +25pt
  - Turn left +25pt
  - Turn right +25pt

**Please familiarise yourselves with these marking steps to ensure the demonstrators can finish marking you in the allocated time**
- [Marking steps](#Marking-steps)
- [Marking checklist](#Marking-checklist)

You will have a **STRICT** 10min time limit to get marked. You may open up the marking checklist, which is a simplified version of the following steps to remind yourself of the marking procedures. 


### Marking steps
#### Step 1:
**Do this BEFORE your lab session**

Zip your whole Week01-02 folder (including the util and pics folder, your modified operate.py with the teleop codes, your readme file) and submit it via the Moodle submission box (according to your lab session). This submission is due by the starting time of the lab session, which means you should **submit your script BEFORE you come to the week 3 lab**. 

**Tip:** Other than a readme file describing your implementation, you may also include a text file in the zip file with a list of commands to use, if you don't know all the commands by heart.

[EDIT] When submitting codes for marking, you don't need to include the venv in the submission folder, as it would be a big zip file to upload/download and takes forever to unzip. You can install the venv in a local directory, and only submit a Week01-02 zip that includes the required scripts for running the teleoperation (operate.py, util/, pics/) on Moodle. Then during marking activate your local venv, cd into the downloaded and unzipped submission and run the live demo

#### Step 2: 
**Do this BEFORE the demonstrator come to mark you**

1. Close all the windows/applications in your dev environment (VM/native/WSL2, whichever one you use)

2. Log in Moodle and navigate to the M1 submission box, so that you are ready to download your submitted code when it is time for your live demo marking

3. Have an **empty** folder named "LiveDemo" ready at the home directory, ie. it is at ```~/LiveDemo/```. This folder should remain open at all time during marking (Does not apply to WSL2 users)

#### Step 3:
**During marking**
1. When the demonstrator start to mark you, download your submitted zip file from Moodle and unzip its content to the "LiveDemo" folder

2. Open another terminal, or new tab in the existing terminal, navigate to the "Week01-02" folder which contains the operate.py script

3. Connect to the physical robot and run the script with ```python3 operate.py --ip 192.168.50.1 --port 8080``` (change the ip address if you need to)

4. Demonstrate the teleoperation and good luck!
---

### Marking checklist
**BEFORE the lab session**
- [ ] Submit your code to Moodle

**BEFORE the marking**
- [ ] Close all programs and folders
- [ ] Log in Moodle and navigate to the submission box
- [ ] Open an empty folder named "LiveDemo" (located at ```~/LiveDemo/```)

**During the marking**
- [ ] Demonstrator will ask you to download your submission and unzip it to "LiveDemo"
- [ ] Connect to the robot and run the operate.py script
- [ ] Demonstrate the teleoperation and good luck!

