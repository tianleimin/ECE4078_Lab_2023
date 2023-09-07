# M4 Evaluation and Marking Instructions 

The arena will contain 10 ArUco markers and 10 fruits&vegs. The full and partial true maps and the shopping list of the 5 targets will be given to you at the start of each lab. The marking arena in each lab session will be slightly different and will be different from the practice arena.

Each team will have a **STRICT** time limit of 15min for the live demo marking, according to this [marking schedule](https://docs.google.com/spreadsheets/d/1bRESqtr_Z1orLkh1-J1UBwSlAibOSyoujpCUq9bQbc8/edit#gid=614009471).

You should first demonstrate the level you are confident with before attempting a more challenging level. You may demonstrate any levels as many times as you want within the 15min live demo time limit.

- [Evaluation](#evaluation)
	- [Level 1: Semi-auto navigation using waypoints](#level-1-semi-auto-navigation-using-waypoints)
	- [Level 2: Autonomous navigation with a known map](#level-2-autonomous-navigation-with-a-known-map)
	- [Level 3: Autonomous navigation with a partially known map](#level-3-autonomous-navigation-with-a-partially-known-map)
- [Marking steps](#marking-steps)
- [Marking checklist](#marking-checklist)

---

## Evaluation
We have divided M4 into 3 levels based on the 3 main components which you could attempt. The levels are
1. Semi-auto navigation with manual inputs (60pts)
2. Autonomous navigation with a known map (20pts)
3. Autonomous navigation with a partially known map (20pts)

**IMPORTANT!!** The following rules apply:
1. Penalty of -2pts for each fruit/veg that the robot collides with

2. Penalty of -2pts for each ArUco marker that the robot collides with

3. Penalty of -5pts each time the robot goes out of the boundary/touches the boundary (+/-1.5m from the origin, in both the x- and y-axis)

4. If you have received three penalties (any of the penalties listed above) during a run, you are disqualified for that run and will receive zero score
	- e.g. zero score for colliding into any object 3 times or collided into 2 objects + out of the boundary once

5. The **end condition of a run** is determined by the robot having attempted to navigate to the 1st-4th targets (0.5m radius to qualify as success), and clearly stopped within **1m of the 5th and last target**. 1m is the acceptable distance to show that a fruit is attempted and is the **intended** stopping location. However, mark for navigating successfully to a fruit will only be given for stopping within the 0.5m radius. Your score for a run will be calculated as:
	- If you decide to stop a run before the end condition or have received 3 penalties, you will get zero score for that run
	- Your robot has to make a visible attempt to try to navigate to all the target fruits, to at least drive pass them within a 1m radius, even if the robot may not accurately stop within 0.5m of the target. As an example, if your robot only reaches the first fruit and stops there, you will not receive any score for that run.

6. The **entire** robot has to be within 0.5m of the target to be considered as successful

7. If the robot reaches the target in the wrong order, you will receive zero score for that run

8. We will review your code to see if you have implemented appropriate algorithms for the levels you have attempted. To gain credit for level 2 or 3, we must find evidence of path planning, or obstacle detection and avoidance (respectively) in your code. Successfully navigating to targets and/or avoiding collisions at these levels by luck will not grant you those marks by default

9. The best run/attempt will be considered as your M4 score

### Level 1: Semi-auto navigation using waypoints
To attempt Level 1, the locations of all the objects in the arena will be given to you in the full groundtruth map. The search order of the target fruits is given in the shopping list.

You are **not allowed to teleoperate** the robot. You can only enter coordinate of the waypoints as input, or you may choose to do so via a GUI. You can input as many waypoints as you want to get to the targets. 

The entire robot needs to be within 0.5m of the target and stop for 2s to indicate that it has found a target fruit/veg before moving onto the next target. You will get 0pt for a target if the robot is not within 0.5m of the target. You should confirm with the demonstrator to check whether the robot is close enough to the target. You will also need to reach the end condition (see above) for a run to qualify for marks.

Each target that you successfully navigate to will give you 12pt if you decide to perform Level 1:
``` 
level1_score = 12 x NumberOfSuccessNav - 2 x CollisionPenalty - 5 x OutOfBoundPenalty
0 ≤ level1_score ≤ 60
```

### Level 2: Autonomous navigation with a known map
To attempt Level 2, the locations of all the objects in the arena will be given to you in the full groundtruth map. The search order of the target fruits is given in the shopping list. You are only allowed to enter a **single command** to launch the navigation program and the robot should perform the task autonomously. 

If you decide to perform Level 2 and have successfully navigated to any number of targets in Level 2 (the run needs to qualify by reaching the end condition as specified above), you will inherit all of Level 1 points, and for each target you have successfully navigated to in Level 2 receive an additional 4pts:
``` 
level2_score = 60 + 4 x NumberOfSuccessNav - 2 x CollisionPenalty - 5 x OutOfBoundPenalty
0 ≤ level2_score ≤ 80
```

### Level 3: Autonomous navigation with a partially known map
To attempt Level 3, the locations of all 10 markers and the 5 targets in the arena will be given to you in the **partial** groundtruth map. The search order of the target fruits is given in the shopping list. The locations of the other 5 objects will **not** be provided in the partial map and you are not allowed to use the full groundtruth map in any part of the implementation to attempt Level 3. If you are found to use the full groundtruth map in your Level 3 implementation you will receive 0pt for M4. You are only allowed to enter a **single command** to launch the navigation program and the robot should perform the task autonomously. 

If you decide to perform Level 3 and have successfully navigated to any number of targets in Level 3 (the run needs to qualify by reaching the end condition), you will inherit all of Level 1 and Level 2 points, and for each target you have successfully navigated to in Level 3 receive an additional 4pts:
``` 
level3_score = 60 + 20 + 4 x NumberOfSuccessNav - 2 x CollisionPenalty - 5 x OutOfBoundPenalty
0 ≤ level3_score ≤ 100
```

---

### Marking instructions
You may open up the marking checklist during the live demo marking, which is a simplified version of the following steps to remind yourself of the marking procedures. 

You MUST follow all the evaluation rules outlined [above](#evaluation), make sure you check out all the rules and understand all of them. 


### Marking steps
#### Step 1:
**Do this BEFORE your lab session**

Zip your implementation and submit via the Moodle submission box (include all scripts that are relevant, such as the wheel and camera calibration parameters, your SLAM component, your trained YOLO detector, custom GUI for waypoint selection, etc). Each group only needs one submission. This submission is due by the starting time of the lab session, which means you should **submit your script BEFORE you come to the lab**. 

**Tips:** 
- You may also include a text file in the zip with a list of commands to use, if you don't know all the commands by heart.
- Practise the marking steps (e.g. unzipping your code and running it) to ensure there are no issues during marking.
- You may update the wheel and camera calibration parameters in the submission at the time of marking. All other scripts in your submission will need to be used as-is.


#### Step 2: 
**Do this BEFORE the demonstrator come to mark your team**

1. Close all the windows/applications

2. Use any team member's account to log in Moodle and navigate to the M4 submission box, so that you are ready to download your submitted code when it's your group's turn to run the live demo

3. Have an **empty** folder named "LiveDemo" ready at the home directory. This folder should remain open at all time during marking

4. Calibrate your robot if needed (you can replace the wheel and camera calibration parameters in the downloaded submission)

5. Connect to eduroam/internet so that you are ready to download your submission from Moodle. Don't connect to the physical robot just yet

6. Activate your Python venv if applicable

#### Step 3:
**During marking**

**Note**: You may attempt any level you want, and you should make it clear to the demonstrator which level you are attempting. Within the 15min marking time limit, you may have as many attempts as you want. The attempt with the highest score will be your final M4 score. 

1. The demonstrator will release the full and partial true maps and the shopping list for each marking arena via Slack at the beginning of each lab session. Note that each lab session will have slightly different marking maps, and the marking maps are different from the practice map provided in the repo. Make sure that the correct true map is used when running your M4 demo

2. When it's your group's turn, go to the marking arena, download your submitted zip file from Moodle and unzip its content to the "LiveDemo" folder. 

3. Place the true map of your marking arena inside your unzipped submission folder. 
    
4. Connect to the robot.

5. Open a terminal, or a new tab in the existing terminal (with the Python venv activated), navigate to the submission folder and run your M4 script by running [auto_fruit_search.py](auto_fruit_search.py) or whichever script(s) you need for the your chosen level to attempt
    - you may take the full or partial true map and the shopping list as the input files, depending on the level that you are attempting

6. For Level 1, you may enter as many waypoints as you want. For Level 2 or 3, you can only input a single command to start the autonomous navigation program
    - We will review your code and you will receive 0pt for M4 if we find that you are teleoperating the robot or used the full true map in Level 3

7. Repeat any level as many times as you want until the time limit

---

### Marking checklist
**BEFORE the lab session**
- [ ] Submit your code to Moodle

**BEFORE the marking**
- [ ] Close all programs and folders
- [ ] Login Moodle and navigate to the submission box
- [ ] Open an empty folder named "LiveDemo"
- [ ] Calibrate the robot if needed
- [ ] Connect your Wifi to eduroam/internet so you are ready to download the submission
- [ ] Activate Python venv if needed

**During the marking** (15min time limit)
- [ ] True maps and shopping list for your session will be released
- [ ] Demonstrator will ask you to download your submission and unzip it to "LiveDemo"
- [ ] Copy the true maps (and the calibration files if you re-calibrated)
- [ ] Connect to robot
- [ ] Run your navigation program, announce to the demonstrator which level you are attempting
- [ ] Enter as many waypoint as you want for level 1, but only a single command for level 2 or 3
- [ ] Run navigation as many times as you want and good luck!

