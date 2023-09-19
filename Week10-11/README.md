# Milestone 5: Integrated System (trial run for final demo)
- [Introduction](#introduction)
- [Tips](#tips)
- Marking: see [M5_marking.md](M5_marking.md)

**Please note that all skeleton codes provided are **for references only** and are intended to get you started. They are not guaranteed to be bug-free either. To get better performance please make changes or write your own codes and test cases. As long as your codes perform the task and generate estimation maps that can be marked according to the evaluation schemes and scripts you are free to modify the skeleton codes.**

---
## Introduction
Milestone 5 is the integration of all the modules you completed in the previous milestones. Integrate your work from M1 to M4 so that the robot can create a map of the arena containing the estimated poses of 10 ArUco markers ([M2: SLAM](../Week03-05/)), 10 objects ([M3: CV](../Week06-07/)), and then perform the grocery shopping task ([M4: navigation](../Week08-09/)). The actual task is very similar to the Level 3 M4 task, which is

**M5 task:** Given a shopping list ([example](../Week08-09/M4_prac_shopping_list.txt)) of 5 targets, your task is to autonomously navigate to the given list of fruits&vegs in order, while avoiding obstacles along the way. The robot should stop within 0.5m radius of the target for 2 seconds before moving onto the next target. The following also applies to M5:
- **A true map will NOT be given in M5**, only the shopping list will be provided indicating which 5 targets are to be reached in which order
- You may teleoperate your robot (M1) to first generate a map of the arena with your own SLAM (M2) and CV (M3) before performing the navigation (M4)
- There will be 10 ArUco markers and 10 objects (5 as navigation targets, 5 as obstacles) in the arena. Similar to M4, the targets on the shopping list will be unique while the obstacles may contain duplicates
- You may choose to perform semi-auto waypoint navigation (same as [M4 Level 1](../Week08-09/M4_marking.md#level-1-semi-auto-navigation-using-waypoints)) to get partial marks if you are struggling to perform a full auto navigation

**The final demo will follow the same procedure**, M5 is also your trial run for final demo to help you identify improvements to be made before the final demo.

**Important note**: 
- As usual, you are not allowed to use true poses of robot or objects during mapping, manually interfere with the robot or the arena, or teleoperate the robot during navigation
- Although the final demo is likely to be the same as M5 (at least the task), minor details/rules/marking scheme are still subject to change
- **PLEASE COMMENT YOUR CODE**. This will speed up the marking process and allow us to provide better feedback if you clearly explain your workflow in your code

---
## Tips
Below are some suggestions on how you may improve your integrated system and live demo performance.

### General remarks
- In the marking arenas of M5 and final demo, there will be 10 ArUco marker blocks and 10 objects. At the starting location (0, 0, 0) the robot will be able to see at least one marker.
- Perform SLAM and object recognition simultaneously, so that after manually driving the robot around the arena once (to save time), you will have both the estimated poses of ArUco markers and the estimated poses of objects
- Make sure you have included everything required to run your demo in the submission. If you can't run the demo from your downloaded submission we can't allow you to run from your local working directory or make changes to your codes. 
- Practice the [marking steps](M5_marking.md#marking-steps) so that you are familiar with it
- Consider having a back-up driver in case the lead person running the demo has any kind of unexpected emergencies

### SLAM
- A [SLAM test case](test/) is added for you to check if your SLAM is implemented correctly
- To calibrate the noise/covariance matrix of the pibot model, you should estimate the values based on what you observed from the tuning and testing phase. You can get a more accurate set of values by calculating the variance for each driving strategy that you use. For example, for driving straight, drive forward 1m for 10 times and record the variance and convert to tick/s. We have provided a [wheel test script](wheel_test.py) to help you tune the variances.
    ```
    if lv and rv == 0: # Stopped
        cov = 0
    elif lv == rv:  # Lower covariance since driving straight is consistent
        cov = 1
    else:
        cov = 2 # Higher covariance since turning is less consistent
    ```

- In [ekf.py](../Week03-05/slam/ekf.py#L129) the second expression in the process noise equation can be commented out since it may add too much noise to the model which accumulates even if the robot is idle. However, you can also add a condition to add this term only when the robot is moving or lower the noise. 
    ```
    Q[0:3,0:3] = self.robot.covariance_drive(raw_drive_meas) #+ 0.01*np.eye(3)
    ```
    
- To test if you implemented derivative_drive and predict correctly, run slam in an empty map (without markers) and record the robot pose. If the robot pose is way off, your calculation is most likely to be wrong. 
- To test if you implemented derivative_drive and predict correctly, run slam with markers and drive around. If RMSE is too high, the calculation for update is most likely to be wrong.

### CV
- Consider making your own test cases to test your CV component without the influence of SLAM accuracy
- Improve your CV's ability to handle occlusion (e.g., an ArUco marker blocking an object)
- Improve your CV's ability to merge multiple observations of the same object vs. seeing duplicates in the obstacles

### Navigation
- If you are taking the semi-auto waypoint navigation approach, improve the map visualization for interactively specifying waypoints during delivery and verifying whether the robot stops inside / outside the 0.5m radius of a target
- During navigation, consider including opportunities for the robot to self-correct its path or reset its location in the arena to prevent errors from accumulating
- Make use of your SLAM component to continuously correct pose estimation and update the path planning
- Make use of visual information to improve the pose estimation and path planning. For example, keeping an object in the centre of the robot's view when driving towards it
