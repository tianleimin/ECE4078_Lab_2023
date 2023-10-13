# Final Demo

## Introduction
The task for the final demo is the same as your trial run in [M5](../Week10-11#introduction).

## Marking
The final demo evaluation schemes are the same as your trial run in [M5](../Week10-11/M5_marking.md#marking-schemes). Please note that the final demo accounts for 60% of your unit mark.

## Marking procedure
The final demo marking procedure is the same as your trial run in [M5](../Week10-11/M5_marking.md#marking-steps).

---

## [NEW] Revised Final Demo marking scheme
We have made the following changes to the marking scheme of your Final Demo compared to M5, please read them carefully:

**1. Demo duration increase**
  - The total duration of your Final Demo is increased to **30min**.
  - We will set up an appointment calendar for teams who intend to perform their Final Demo during the Swot Vac to choose when to perform their Final Demo. The Final Demo marking sessions will be running between 10am and 5pm on Monday, Tuesday, and Wednesday of Swot Vac (23, 24, 25 Oct). All teams performing their Final Demo during Swot Vac will need to submit their codes on [Moodle](https://lms.monash.edu/mod/assign/view.php?id=12377512) **before 9:30am on 23 Oct**, and submit their generated maps on [Moodle](https://lms.monash.edu/mod/assign/view.php?id=12377509) at the end of their Final Demo.
  - Within this 30min, you need to download and unzip your submitted codes, perform the demo, reset arena in between runs if needed, and submit your generated SLAM and targets maps with the required format and file names.

**2. Navigation marking scheme change**
  - The 60pt navigation mark will now consist of:
    - 5pt if there is evidence in your code submission that your robot can perform waypoint based semi-auto navigation
    - 10pt for semi-auto navigation, with 2pt awarded for each of the 5 targets you successfully navigate to (within 0.5m radius) in the order of the shopping list in a qualified run
    - 5pt if there is evidence in your code submission that your robot can perform full auto navigation using any path planning algorithms
    - 40pt for full auto navigation, with 8pt awarded for each of the 5 targets you successfully navigate to (within 0.5m radius) in the order of the shopping list in a qualified run
  - If you have achived a qualified full auto navigation run, you can receive the 15pt associated to semi-auto navigation without needing to perform a semi-auto navigation run

**3. Qualified navigation run and penalty scheme changes**
  - To achieve a qualified navigation run, you need to navigate to **3 out of the 5** targets in the order specified in the shopping list, and for each of them stop within 1m of the target (you will only get marks for successfully navigating to a target if the robot is within 0.5m of that target).
  - After the robot has navigated to 3 targets, you may stop the run at any time either manually or as your program stops by itself and keep the marks you've earned so far. For example, if your robot successfully navigated to the 1st and 2nd target without penalty, and you stopped it within 1m radius but outside of 0.5m radius of the 3rd target in a full auto run, you will receive 16pt for that run (a qualified run with 8pt awarded for each of the two successful targets).
  - The collision and out-of-arena-boundary penalties are changed to:
    - 1st and 2nd collisions OR out-of-arena-boundary -2pt each
    - 3rd and 4th collisions OR out-of-arena-boundary -5pt each
    - 5th time a collision OR out-of-arena-boundary happens the run will terminate. However, this does not neccessarily disqualify that run and you will keep the marks already earned in that run. For example, if the 5th penalty happend when your robot is making its way from the 4th target to the 5th target, you will still get the marks associated with the first 4 targets minus the penalties if your robot was within 1m of the first 3 targets.
    - You are not allowed to manually reset the robot or the arena during a semi or full auto navigation run, such as moving aside a marker block that the robot gets stuck behind.
    - Manual interference to code execution during a full auto run after the run is launched with your one command will immediately terminate that run
    - The penalty will not exceed the total mark of the run: 0 ≤ (score of a semi auto run) ≤ 10pt, 0 ≤ (score of a full auto run) ≤ 40pt

**4. Individual contributions to team and mark scaling**
  - The 3rd and last ITP survey will be open from 10am 16 Oct to 6pm 20 Oct. The results will be used to inform the individual scaling factors applied to M5 and Final Demo's team scores.
  - We may conduct individual interview and code reviews as part of the final assessment to understand an individual's contribution to the team. The interview results may be used to adjust the individual scaling factor. The teaching team will email individuals during Swot Voc to arrange this interview if needed.

**5. Other**
  - The Final Demo will be video recorded. We may record the arena, the robot's behaviours inside the arena, your computer screens or keyboard actions. We will not record any people or faces. Any accidental recordings of personal information irrelevant to the Final Demo or the unit will be deleted.
  - We will prepare a small number of back-up robots with calibrated wheel and camera parameters ready to use with them. During the Final Demo, if a team has unexpected hardware issues they may switch to use these back-up robots. While switching and reconnecting to the back-up robot the demo timer will be paused.

---

## [NEW] Further clarifications and FAQs
Below are issues that we have seen during teams' M5 runs which may be helpful to address:
1. HAVE A BACK-UP PLAN!!! For example, implement a command line argument for switching between running semi or full auto navigation in case the full auto navigation crashes on the day. Also have a back-up driver and laptop.
2. Parameter tuning: check your wheel and camera parameters, YOLO confidence, SLAM covariance, radius around markers and objects for creating occupancy map, etc. We recommand recalibrating the wheel and camera during the Week 12 labs and check to make sure that your calibrated parameters are close to the [default parameters](../Week03-05/calibration/param/).
3. Try different map layouts and pay attention to when collisions might happen due to path finding not optimised or occupancy map radius not being able to handle inaccurate maps
4. Make sure to submit the right version of your codes containing all required components. Test to ensure that the codes work as expected.
5. We are still seeing map syntax and naming errors in the submitted 'slam.txt' and 'targets.txt' which might result in 0pt mapping scores. Please make sure to check the maps generated and submit the generated maps that you want to be marked on.
6. Check if your EKF is correclty integrated and that your SLAM is correctly implemented.
7. Some groups had their generated SLAM map rotated or flipped on the x/y axis, please check this with practice arenas and sim maps. With the robot's camera facing left when positioned in the middle of an arena, the left half of the arena will have positive x coordinates, and the bottom half of the arena will have positive y coordinates
8. Practice your runs and discuss driving strategies with your teammates to reduce operator errors (e.g., pressing wrong buttons or giving wrong commands)
9. In a navigation run, distance of the 0.5m radius for successful navigation and 1m radius for qualified navigation is measured from the centre of the target, and the **entire** robot has to be within this radius
