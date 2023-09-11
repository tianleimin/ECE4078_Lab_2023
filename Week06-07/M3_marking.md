# Milestone 3: Marking
- [Evaluation](#evaluation)
- [Live demo marking steps](#live-demo-marking)
- [Live demo marking checklist](#marking-checklist)

---

## Evaluation
Your M3 score consists of two parts: the accuracy of your YOLO detector for classifying the type of a target object, and the accuracy of your estimated map containing the location of the 10 targets in the arena.

The performance of your YOLO detector worths 20pts. You'll demonstrate your trained YOLO model on a set of 10 testing images, each image containing one fruit/veg. For each testing image, if the predicted class label is correct, you'll receive 2pts. The detector evaluation is run on your computer without using the robot (pass the set of marking images to your trained YOLO model and show a visualisation of the predicted bounding boxes and labels):

~~~
detector_score = 2 x NumberOfCorrectPredictions
0 ≤ detector_score ≤ 20
~~~

[mapping_eval.py](mapping_eval.py) is used to evaluate performance of your target pose estimation. It requires both the SLAM map ```lab_output/slam.txt``` and the targets map ```lab_output/targets.txt```. Run 
```
python mapping_eval.py
```
This computes the Euclidean distance between each target's true location in "true_map.txt" and its closest estimation in "lab_output/targets.txt" (the estimation error). It also generates an estimation on your SLAM's performance as error in SLAM will impact your target pose estimation performance. The SLAM map is used to align the object pose estmation as well.

"targets.txt" should contain the poses of all 10 fruits and vegs in the arena. **Make sure your modified "TargetPoseEst.py" generates a "targets.txt" file that is in the same format as the [given example output](lab_output/targets.txt)**. Generating the estimation map in a wrong format may result in it not being compatible with the evaluation scripts and thus getting 0pt for M3 target_est_score.

Your target_est_score is calculated by summing the pose estimate error of all 10 targets, with the error of each target having an upper bound of 1.0 and lower bound of 0.025 

There is also a 5pt collision penalty for each object / ARUCO marker that you collide with or each time your robot run out of the arena's boundries during a demo run, with a maximum of 3 collisions/out-of-boundary allowed in a run: 

~~~
target_score[object] = (1 - estimation_error[object])/(1-0.025) x 8
target_est_score = sum(target_score) - 5 x NumberOfCollisions
0 ≤ target_est_score ≤ 80
~~~

Your M3 score is the sum of the detector performance score and the target pose estimation performance score:

~~~
M3_score = detector_score + target_est_score
0 ≤ M3_score ≤ 100
~~~

---

## Live demo marking
You have a **20min time limit** for M3 live demo marking, this includes time taken to download and unzip your submissions, to demonstrate your detector on a set of marking images, to demonstrate your target pose estimation by driving the robot around a marking arena, and to submit the generate maps.

### Marking steps
#### Step 1:
**Do this BEFORE your lab session**
Zip your M3 implementation and submit to the Moodle. Each group only needs one submission. This submission is due by the starting time of the lab session, which means you should **submit your script BEFORE you come to the lab**:
1. Include ALL scripts required for performing the fruit/veg detection and pose estimation in the zip, including the trained YOLO .pt model
2. Please DO NOT submit your training dataset or the python venv. If your trained YOLO model file is too big to upload to Moodle, please upload it to your Google Drive and provide a link to it in your README.txt
3. You cannot change the submitted scripts when running the live demo, except for replacing the [calibrated wheel and camera parameters](../Week03-05/calibration/param/), so make sure you submitted the right implementation version.
4. Your code may take in command line inputs unrelated to the arena / objects / markers / robot status. For example, your code may take in a run ID for automatically renaming the generated slam.txt and targets.txt to make sure the pairs are aligned.
5. We may review your code or ask you to explain your implementation.

**Tips:** 
- You can write a loop for the YOLO detector to take in all images in a directory instead of manually changing the image file name to speed things up.
- You may also include a text file in the zip with a list of commands to use, if you don't know all the commands by heart.
- **Please practise** the marking steps (eg. unzipping your code and running it) to ensure there are no issues during marking.
- Make sure your robot is fully charged before coming to the lab session.


#### Step 2: 
**Do this BEFORE the demonstrator marks your team**

1. Close all the windows/applications in your development environment (VM/native/WSL2, whichever one you use)

2. Use any team member's account to log in Moodle and navigate to the M3 submission box, so that you are ready to download your submitted code when the demonstrator arrives

3. Have an **empty** folder named "LiveDemo" ready in your home directory, ie. it is at ```~/LiveDemo/```. This folder should remain open at all time during marking (Does not apply to WSL2 users)

4. Please turn on your robot before the demonstrator comes to you. DO NOT connect to its hotspot yet. 

#### Step 3:
**During marking**
Note: within the 20min marking time limit, you may perform detection of the marking images or target pose estimation as many times as you want. The attempt with the highest score will be your final detector_score and target_est_score. 

1. The demonstrator will set up the marking arenas containing the 10 ARUCO markers and 10 target objects at the beginning of the session. Note that each lab session will get slightly different map layouts. A set of 10 detector marking images will also be provided (each image containing one target object, a different set will be used for each lab session). **You are not allowed near the marking arena unless it is your team's turn for marking**

2. When the demonstrator ask you to come to the marking arena, download your submitted zip file from Moodle and unzip its content to the "LiveDemo" folder.

3. Activate your Python venv, then navigate to the "Week06-07" folder in your unzipped submission which contains the operate.py script

4. Demonstrate your object detector by running your YOLO detector with the set of 10 marking images and show a visualisation of the bounding box and object label of each image. The demonstrator will record the number of correctly classified images for the detector_score. You may rerun this step as many times as you want within your marking time limit.

5. Connect to the robot and run the script with ```python3 operate.py --ip 192.168.50.1 --port 8080```

6. Demonstrate target pose estimation by **strategically** driving the robot around arena to observe all 10 targets, and then run TargetPoseEst.py to generate a map based on the observations
    - You may stop a run at anytime 
    - You may rerun this step as many times as you want within your marking time limit. **Remember that the map submission has to be done within the time limit**, so make sure you leave enough time for the submission. The 20min marking timer also continues when you are getting ready in between runs (resetting the robot to the start location and resetting any markers or objects that may have moved due to collision).
    - A visible effort has to be made to try and locate all 10 objects.
    - **The maximum number of collision/out-of-bound allowed in a run is 3 times**. The fourth time your robot collides with a marker or an object or run out of arena boundaries, you will be asked to terminate that run and restart a new run if time permits and if you want to
    - Note that the provided skeleton code saves the map under the same name (i.e., the later run would replace map generated by the earlier run), so you'll need to manually rename the maps in between runs, or modify the map-saving codes when developing your solution.
    - You are not allowed to change anything in the downloaded submission of your implementation, except for the camera and wheel calibration parameters. Your code can also take in arguments at the time of execution. However, you are NOT allowed to give inputs related to the arena setup or object locations/types as command line arguments, an extreme example would be typing in the whole map manually as command line arguments (please don't). 

7. Submit the generated slam.txt and targets.txt maps (make sure SLAM and target maps are matching pairs for each run) as a zip to the Moodle map submission box for target_est_score marking

---

### Marking checklist
**BEFORE the lab session**
- [ ] Submit your code to Moodle
- [ ] Charge the robot

**BEFORE the marking**
- [ ] Close all programs and folders
- [ ] Log in Moodle and navigate to the submission box
- [ ] Open an empty folder named "LiveDemo" (located at ```~/LiveDemo/```)
- [ ] Turn on the robot (DO NOT connect to its hotspot yet)

**During the marking** (20min time limit)
- [ ] Demonstrator will ask you to download your submission and unzip it to "LiveDemo"
- [ ] Demonstrate detector performance by running YOLO with the marking image set
- [ ] Connect to the robot
- [ ] Demonstrate target pose estimation (rename the ```lab_output/slam.txt``` and ```lab_output/targets.txt``` files by run ID, max 3 collisions/out-of-bound per run) and good luck!
- [ ] zip and submit the map(s) to Moodle
