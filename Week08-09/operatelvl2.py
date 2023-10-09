# teleoperate the robot, perform SLAM and object detection

import os
import sys
import time
import cv2
import numpy as np
import math
import json
import copy

#from RRT import *
from rrt2 import *
from Obstacle import *
from a_star import AStarPlanner
# import utility functions
sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi    # access the robot
import util.DatasetHandler as dh    # save/load functions
import util.measure as measure      # measurements
import pygame                       # python package for GUI
import shutil                       # python package for file operations

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import YOLO components 
#from YOLO.detector import Detector


class Operate:
    def __init__(self, args):
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)

        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = PenguinPi(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length=0.07)  # size of the ARUCO markers

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion': [0, 0],
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on =True # False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        self.pred_notifier = False
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.aruco_img = np.zeros([240, 320, 3], dtype=np.uint8)
        #self.detector_output = np.zeros([240, 320], dtype=np.uint8)
        """
        if args.yolo_model == "":
            self.detector = None
            self.yolo_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            self.detector = Detector(args.yolo_model)
            self.yolo_vis = np.ones((240, 320, 3)) * 100
        self.bg = pygame.image.load('pics/gui_mask.jpg')
        """
    def get_robot_pose(self):
        states = self.ekf.get_state_vector()
        #print(f"robot pose: {states}")
        return states[0:3, :]
        
    def read_true_map(self, fname):
        """Read the ground truth map and output the pose of the ArUco markers and 5 target fruits&vegs to search for

        @param fname: filename of the map
        @return:
            1) list of targets, e.g. ['lemon', 'tomato', 'garlic']
            2) locations of the targets, [[x1, y1], ..... [xn, yn]]
            3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
        """
        with open(fname, 'r') as fd:
            gt_dict = json.load(fd)
            fruit_list = []
            fruit_true_pos = []
            aruco_true_pos = np.empty([10, 2])

            # remove unique id of targets of the same type
            for key in gt_dict:
                x = np.round(gt_dict[key]['x'], 1)
                y = np.round(gt_dict[key]['y'], 1)

                if key.startswith('aruco'):
                    if key.startswith('aruco10'):
                        aruco_true_pos[9][0] = x
                        aruco_true_pos[9][1] = y
                    else:
                        marker_id = int(key[5]) - 1
                        aruco_true_pos[marker_id][0] = x
                        aruco_true_pos[marker_id][1] = y
                else:
                    fruit_list.append(key[:-2])
                    if len(fruit_true_pos) == 0:
                        fruit_true_pos = np.array([[x, y]])
                    else:
                        fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)
                        
                    #print(f'fruit: {fruit_list[-1]} at {fruit_true_pos}')

            #return (1), (2), (3)
            return fruit_list, fruit_true_pos, aruco_true_pos


    def read_search_list(self):
        """Read the search order of the target fruits

        @return: search order of the target fruits
        """
        search_list = []
        with open('search_list.txt', 'r') as fd:
            fruits = fd.readlines()

            for fruit in fruits:
                search_list.append(fruit.strip())

        return search_list


    def print_target_fruits_pos(self, search_list, fruit_list, fruit_true_pos):
        """Print out the target fruits' pos in the search order

        @param search_list: search order of the fruits
        @param fruit_list: list of target fruits
        @param fruit_true_pos: positions of the target fruits
        """

        print("Search order:")
        n_fruit = 1
        for fruit in search_list:
            for i in range(len(fruit_list)): # there are 5 targets amongst 10 objects
                if fruit == fruit_list[i]:
                    print('{}) {} at [{}, {}]'.format(n_fruit,
                                                    fruit,
                                                    np.round(fruit_true_pos[i][0], 1),
                                                    np.round(fruit_true_pos[i][1], 1)))
            n_fruit += 1

    '''
    # wheel control
    def control(self):
        if args.play_data:
            lv, rv = self.pibot.set_velocity()
        else:
            lv, rv = self.pibot.set_velocity(
                self.command['motion'])
        if self.data is not None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        # running in sim
        if args.ip == 'localhost':
            drive_meas = measure.Drive(lv, rv, dt)
        # running on physical robot (right wheel reversed)
        else:
            drive_meas = measure.Drive(lv, -rv, dt)
        self.control_clock = time.time()
        return drive_meas
    '''
    
    
    
    
    
    
    
    
    def drive_to_waypoint(self, waypoint, robot_pose):
        '''
            Function that implements the driving of the robot
        '''
        
        # Read in baseline and scale parameters
        datadir = "calibration/param/"
        scale = np.loadtxt("{}scale.txt".format(datadir), delimiter=',')
        
        # Read in the waypoint values
        waypoint_x = waypoint[0]
        waypoint_y = waypoint[1]
        
        # Read in robot pose values
        robot_pose_x = robot_pose[0]
        robot_pose_y = robot_pose[1]
        
        # Wheel ticks in m/s
        wheel_ticks = 30

        target_angle = operate.turn_to_waypoint(waypoint, robot_pose)

        # Drive straight to waypoint
        distance_to_waypoint = math.sqrt((waypoint_x - robot_pose_x)**2 + (waypoint_y - robot_pose_y)**2)
        drive_time = abs(float((distance_to_waypoint) / (wheel_ticks*scale))) 

        print("Driving for {:.2f} seconds".format(drive_time))
        operate.motion_controller([1,0], wheel_ticks, drive_time)
        print("Arrived at [{}, {}]".format(waypoint[1], waypoint[0]))
        
        update_robot_pose = [waypoint[0], waypoint[1], target_angle]
        return update_robot_pose 

    def turn_to_waypoint(self, waypoint, robot_pose):
        '''
            Function that implements the turning of the robot
        '''
        # Read in baseline and scale parameters
        datadir = "calibration/param/"
        scale = np.loadtxt("{}scale.txt".format(datadir), delimiter=',')
        baseline = np.loadtxt("{}baseline.txt".format(datadir), delimiter=',')

        # Read in the waypoint values
        waypoint_x = waypoint[0]
        waypoint_y = waypoint[1]
        
        # Read in robot pose values
        robot_pose_x = robot_pose[0]
        robot_pose_y = robot_pose[1]
        robot_pose_theta = robot_pose[2]
        
        # Wheel ticks in m/s
        wheel_ticks = 30

        # Calculate turning varibles
        turn_time = 0
        theta_target = math.atan2(waypoint_y - robot_pose_y, waypoint_x - robot_pose_x) # angle from robot's current position to the target waypoint
        theta_delta = theta_target - robot_pose_theta # How far the robot must turn from current pose
        
        if theta_delta > math.pi:
            theta_delta -= 2 * math.pi
        elif theta_delta < -math.pi:
            theta_delta += 2 * math.pi

        # Evaluate how long the robot should turn for
        turn_time = float((abs(theta_delta)*baseline)/(2*wheel_ticks*scale))
        print("Turning for {:.2f} seconds".format(turn_time))
        
        if theta_delta == 0:
            print("No turn")
        elif theta_delta > 0:
            operate.motion_controller([0,1], wheel_ticks, turn_time)
        elif theta_delta < 0:
            operate.motion_controller([0,-1], wheel_ticks, turn_time)
        else:
            print("There is an issue with turning function")
            
        return theta_delta # delete once we get the robot_pose working and path plannning
        
        
        
    def motion_controller(self, motion, wheel_ticks, drive_time):
        lv,rv = 0.0, 0.0
        
        if not motion == [0,0]:
            if motion[0] == 0:  # Turn
                lv, rv = ppi.set_velocity(motion, tick=wheel_ticks, time=drive_time)
            else:   # Drive forward
                lv, rv = ppi.set_velocity(motion, tick=wheel_ticks, time=drive_time)
            
            # A good place to add the obstacle detection algorithm
            
            # Run SLAM Update Sequence
            operate.take_pic()
            drive_meas = measure.Drive(lv,rv,drive_time)
            operate.update_slam(drive_meas)
            #operate.record_data()
            #operate.save_image()
            #operate.detect_target()
        
             
        
    '''
    def drive_to_point(self, waypoint, robot_pose, args):
        print(f"Current pose: {robot_pose}")
        #import camera and baseline calibration parameters
        #fileS = "calibration/param/scale.txt"
        #scale = np.loadtxt(fileS, delimiter='')
        datadir,ip_ = args.calib_dir, args.ip
        
        scale = np.loadtxt("{}scale.txt".format(datadir), delimiter=',')
        
        #fileB = "calibration/param/baseline.txt"
        baseline = np.loadtxt("{}baseline.txt".format(datadir), delimiter=',')

        # Wheel speed in tick
        wheel_vel = 30  # tick   
        # Find wheel speed in m/s
        left_wheel_speed = wheel_vel * scale
        right_wheel_speed = wheel_vel * scale
        
        # Linear and Angular velocity
        linear_velocity = (right_wheel_speed + left_wheel_speed) / 2.0 # instead of using wheel_vel use linear_vel      
        #angular_velocity = (right_wheel_speed + left_wheel_speed) / baseline
        
        #print(f'linear vel: {linear_velocity}, angular vel: {angular_velocity}')
        
        # Determine the angle the robot needs to turn to face waypoint
        #current_angle = robot_pose[2][0]
        current_angle = robot_pose[2]
        target_angle = math.atan2(waypoint[1] - robot_pose[1], waypoint[0] - robot_pose[0])
        angle_to_turn = target_angle - current_angle

        print(f'Current angle: {current_angle}, Target angle: {target_angle}, angle to turn: {angle_to_turn}')
        
        # Ensure the turn angle is within the range of -π to π (or -180 degrees to 180 degrees)
        if angle_to_turn >= math.pi:
            angle_to_turn -= 2 * math.pi
            #self.command[0,-1]
        elif angle_to_turn < -math.pi:
            angle_to_turn += 2 * math.pi
            #self.command[0,1]

        operate.control_clock=time.time()
        #turn_time = abs(angle_to_turn / (2*np.pi * baseline))
        #turn_time = abs(angle_to_turn / (np.pi / 4))
        turn_time = abs(baseline*angle_to_turn*0.5) / (scale*wheel_vel)
        
        #print("Turning for {:.2f} seconds".format(turn_time))
        print(f'turning for {turn_time}')
        #ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
        self.command['motion'] = [0, 1]        
        turn_time += time.time()
        while time.time() <= turn_time:
            #print("turning...")
            #self.take_pic()
            drive_meas = self.control()
            self.update_slam(drive_meas)
            #self.record_data()
            #self.save_image()
            
        
        # Drive straight to the waypoint
        #distance_to_waypoint = math.sqrt((waypoint[0] - robot_pose[0][0])**2 + (waypoint[1] - robot_pose[1][0])**2)
        distance_to_waypoint = math.sqrt((waypoint[0] - robot_pose[0])**2 + (waypoint[1] - robot_pose[1])**2)

        print(linear_velocity)
        #drive_time = abs((distance_to_waypoint) / wheel_vel) # could minus 0.5m from waypoint to get to radius of 0.5
        
        drive_time = abs((distance_to_waypoint) / (wheel_vel*scale))
        
        print(f'Distance to drive: {distance_to_waypoint}')


        print("Driving for {:.2f} seconds".format(drive_time))
        #ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
        self.command['motion'] = [1,0]
        
        drive_time += time.time()
        while time.time() <= drive_time:
            #self.take_pic()
            drive_meas = self.control()
            self.update_slam(drive_meas)
            #self.record_data()
            #self.save_image()
            #print("---------------------------")
            #print(self.get_robot_pose())
            #print("----------------------------")
            
        self.command['motion'] = [0,0]
        self.control()
        time.sleep(2)
        print("Arrived at [{}, {}]\n".format(waypoint[0], waypoint[1]))
    '''
    
    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()

        if self.data is not None:
            self.data.write_image(self.img)

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on:  # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)

    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            # need to convert the colour before passing to YOLO
            yolo_input_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)

            self.detector_output, self.yolo_vis = self.detector.detect_single_image(yolo_input_img)

            # covert the colour back for display purpose
            self.yolo_vis = cv2.cvtColor(self.yolo_vis, cv2.COLOR_RGB2BGR)

            # self.command['inference'] = False     # uncomment this if you do not want to continuously predict
            self.file_output = (yolo_input_img, self.ekf)

            # self.notification = f'{len(self.detector_output)} target type(s) detected'

    # save raw images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)

    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False
        # save inference with the matching robot pose and detector labels
        if self.command['save_inference']:
            if self.file_output is not None:
                # image = cv2.cvtColor(self.file_output[0], cv2.COLOR_RGB2BGR)
                self.pred_fname = self.output.write_image(self.file_output[0],
                                                          self.file_output[1])
                self.notification = f'Prediction is saved to {operate.pred_fname}'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False

    # paint the GUI            
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(320, 480 + v_pad),
                                            not_pause=self.ekf_on)
        canvas.blit(ekf_view, (2 * h_pad + 320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view,
                                position=(h_pad, v_pad)
                                )

        # for target detector (M3)
        detector_view = cv2.resize(self.yolo_vis, (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view,
                                position=(h_pad, 240 + 2 * v_pad)
                                )

        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2 * h_pad + 320, v_pad))
        self.put_caption(canvas, caption='Detector',
                         position=(h_pad, 240 + 2 * v_pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        notifiation = TEXT_FONT.render(self.notification,
                                       False, text_colour)
        canvas.blit(notifiation, (h_pad + 10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain) % 2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2 * h_pad + 320 + 5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)

    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                            False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1] - 25))

    # keyboard teleoperation, replace with your M1 codes if preferred        
    def update_keyboard(self):
        for event in pygame.event.get():
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'][0] = min(self.command['motion'][0] + 1, 1)
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'][0] = max(self.command['motion'][0] - 1, -1)
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'][1] = min(self.command['motion'][1] + 1, 1)
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'][1] = max(self.command['motion'][1] - 1, -1)
            # stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            # save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm += 1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused'
            # run object detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['inference'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()


    def generate_paths(self, fruits_list, aruco_list, search_list):
        #getting index of fruits to be searched
        #fruit_list_dict = dict(zip(self.fruit_list,range(len(self.fruit_list))))
        #all_fruits = [x for x in range(len(self.fruit_list))]
        #search_fruits = [fruit_list_dict[x] for x in self.search_list]
        #other_fruits = list((set(all_fruits) | set(search_fruits)) - (set(all_fruits) & set(search_fruits)))

        #adding markers as obstacles
        obstacles = []
        #for x,y in self.aruco_true_pos:
        for x,y in aruco_list:
            obstacles.append([x + 1.5, y + 1.5])

        #adding other fruits as obstacles
        for x,y in fruits_list:
            #x,y = self.fruit_true_pos[idx]
            
            obstacles.append([x + 1.5, y + 1.5])

        #printing search fruits location
        for idx in range(len(search_list)):
            print(f' {search_list[idx]} at {fruits_list[idx]}')

        radius = 0.25
        radius_success = False
        while not radius_success:
            try:
                all_obstacles = generate_path_obstacles(obstacles, radius) #generating obstacles

                #starting robot pose and empty paths
                start = np.array([0,0]) + 1.5
                paths = []
                print("New path generated")


                for idx in range(len(search_list)):
                    success = False
                    method = 1
                    linear_offset = 0.3
                    while not success:
                        location = copy.deepcopy(fruits_list[idx])
                        print(f'Location: {location}')
                        if method == 1:
                            offset = 0.2
                            # Stop in front of fruit
                            if location[0] > 0 and location[1] > 0:
                                location -= [offset, offset]
                            elif location[0] > 0 and location[1] < 0:
                                location -= [offset, -offset]
                            elif location[0] < 0 and location[1] > 0:
                                location -= [-offset, offset]
                            else:
                                location += [offset, offset]
                        elif method == 2:
                            if location[1] > 0:
                                location -= [0, linear_offset]
                            else:
                                location += [0, linear_offset]
                        elif method == 3:
                            if location[0] > 0:
                                location -= [linear_offset,0]
                            else:
                                location += [linear_offset,0]
                        elif method == 4:
                            if location[1] > 0:
                                location += [0, linear_offset]
                            else:
                                location -= [0, linear_offset]
                        elif method == 5:
                            if location[0] > 0:
                                location += [linear_offset,0]
                            else:
                                location -= [linear_offset,0]
                        else:
                            break

                        goal = np.array(location) + 1.5

                        try:
                            rrtc = RRT(start=start, goal=goal, width=3, height=3, obstacle_list=all_obstacles,
                                expand_dis=1, path_resolution=0.1)
                            path = rrtc.planning()[::-1] #reverse path
                            success = True
                            print("Success!")
                        except:
                            print(f"{self.fruit_list[idx]} Failed")
                            method += 1

                    print("printing path")
                    #printing path
                    for i in range(len(path)):
                        x, y = path[i]
                        path[i] = [x - 1.5, y - 1.5]
                    print(f'The path is {path}')

                    #adding paths
                    paths.append(path)
                    start = np.array(goal)
                self.paths = paths
                radius_success = True
            except:
                #self.radius -= 0.05
                radius -= 0.05
                print(f"Radius reduced to {radius}")
                
        return paths
'''
    def path_planning(self,search_order):
        fileB = "calibration/param/baseline.txt"
        robot_radius = np.loadtxt(fileB, delimiter=',')*2 # robot radius = baseline of the robot/2.0
        robot_radius = 0.2
        robot_pose = operate.get_robot_pose() # estimate the robot's pose
        print("Search order is:", search_order)
        sx,sy = float(robot_pose[0]),float(robot_pose[1]) # starting location
        # gx,gy = fruits_true_pos[search_order][0],fruits_true_pos[search_order][1] # goal position

        for i in range(3): # to get the correct fruit idx based on the search list
            if search_list[search_order] == fruits_list[i]:
                gx,gy = fruits_true_pos[i][0],fruits_true_pos[i][1] # goal position

        print("starting loation is: ",sx,",",sy)
        print("ending loation is: ",gx,",",gy)
        
    #--------------------------------------- Using AStar-------------------------------------#
        grid_size = 0.20

        a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
        rx, ry = a_star.planning(sx, sy, gx, gy)
    #--------------------------------------- Using AStar-------------------------------------#
        return rx,ry
    
    def initialise_space(self,fruits_true_pos,aruco_true_pos,search_order):
        ox,oy=[],[] # obstacle location

        # to get the fruit idx based on the search list
        for i in range(3):
            if search_list[search_order] == fruits_list[i]:
                search_idx = i
        # define the obstacle location
        for i in range(3):
            if i == search_idx: # do not include the current fruit goal as obstacle
                continue
            ox.append(fruits_true_pos[i][0])
            oy.append(fruits_true_pos[i][1])
        for i in range(10):
            ox.append(aruco_true_pos[i][0])
            oy.append(aruco_true_pos[i][1])

        print("Number of obstacle is : ",len(ox))
        return ox,oy
        '''
# main loop
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--map", type=str, default='M4_true_map_full.txt')
    #parser.add_argument("--map", type=str, default='lab_out/targets.txt')
    
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip,args.port)
    operate = Operate(args)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = operate.read_true_map(args.map) #list of fruits names, locations of fruits, locations of aruco markers
    search_list = operate.read_search_list() #inputted ordered list of fruits to search 
    operate.print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    print(f'fruits_true_pos: {fruits_true_pos}')
    print(f'search list: {search_list}')
    
    # Define starting and robot pose 
    start = np.array([0.0, 0.0])
    robot_pose =  np.array([0.0, 0.0, 0.0])
    waypoint = [0.0,0.0]
    
    
    ################Run SLAM####################
    n_observed_markers = len(operate.ekf.taglist)
    if n_observed_markers == 0:
        if not operate.ekf_on:
            print('SLAM is running')
            operate.ekf_on = True
        else:
            print('> 2 landmarks is required for pausing')
    elif n_observed_markers < 3:
        print('> 2 landmarks is required for pausing')
    else:
        if not operate.ekf_on:
            operate.request_recover_robot = True
        operate.ekf_on = not operate.ekf_on
        if operate.ekf_on:
            print('SLAM is running')
        else:
            print('SLAM is paused')
    ###########################################


    ################Generate Paths####################
    paths = operate.generate_paths(fruits_true_pos, aruco_true_pos, search_list)
    print(f'--------------Final path is {paths}') 
    ###########################################

    
    ############Drive Robot######################
    for path in paths:
        #ignore the starting point (duplicates)
        print(f"Current goal: {path[-1]}")
        for i in range(1, len(path) - 1):
            wp = path[i]
            print(f'Current wp: {wp}')
            robot_pose = operate.get_robot_pose()
            robot_pose = np.array(robot_pose[0], robot_pose[1], robot_pose[2])
            print("Robot Pose:", robot_pose)
            operate.drive_to_waypoint(wp, robot_pose)
        
        # Implement a delay of 2 seconds and update SLAM
        print("Initiating the Delay of 2 seconds")
        operate.motion_controller([0,0],0.0,2)
    ###########################################

sys.exit()