# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import math
import time
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from collections import deque
from random import random
import ast

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
from pibot import PenguinPi
import measure as measure

from rrt import RRTC


def read_true_map(fname):
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

        #return (1), (2), (3)
        return fruit_list, fruit_true_pos, aruco_true_pos


def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
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


# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# note that this function requires your camera and wheel calibration parameters from M2, and the "util" folder from M1
# fully automatic navigation:
# try developing a path-finding algorithm that produces the waypoints automatically
# turn_time = 
# Convert to m/s
#left_speed_m = left_speed * self.wheels_scale
#right_speed_m = right_speed * self.wheels_scale
# angular_vel = (right_speed_m - left_speed_m) / self.wheels_width
# turn_time = angle_to_turn / angular_vel

def motion_control(waypoint, robot_pose, radius):
    #import camera and baseline calibration parameters
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter='')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter='')
    
    # Wheel speed in tick
    wheel_vel = 30  # tick   
    # Find wheel speed in m/s
    left_wheel_speed = wheel_vel * scale
    right_wheel_speed = wheel_vel * scale
    
    # Linear and Angular velocity
    linear_velocity = (right_wheel_speed + left_wheel_speed) / 2.0 # instead of using wheel_vel use linear_vel      
    angular_velocity = (right_wheel_speed + left_wheel_speed) / baseline
    
    # Determine the angle the robot needs to turn to face waypoint
    current_angle = robot_pose[2]
    target_angle = math.atan2(waypoint[1] - robot_pose[1], waypoint[0] - robot_pose[0])
    angle_to_turn = target_angle - current_angle

    # Ensure the turn angle is within the range of -π to π (or -180 degrees to 180 degrees)
    if angle_to_turn >= math.pi:
        angle_to_turn -= 2 * math.pi
        turn_direction = -1
    elif angle_to_turn < -math.pi:
        angle_to_turn += 2 * math.pi
        turn_direction = 1

    turn_time = abs(angle_to_turn / angular_velocity)
    print("Turning for {:.2f} seconds".format(turn_time))
    ppi.set_velocity([0, turn_direction], turning_tick=wheel_vel, time=turn_time)

    # Drive straight to the waypoint
    distance_to_waypoint = math.sqrt((waypoint[0] - robot_pose[0])**2 + (waypoint[1] - robot_pose[1])**2)
    drive_time = abs((distance_to_waypoint-radius) / linear_velocity) # could minus 0.5m from waypoint to get to radius of 0.5

    print("Driving for {:.2f} seconds".format(drive_time))
    ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)


def drive_to_point(waypoint, robot_pose):
    # Determines a set distance to travel from the target
    desired_radius = 0.5
    
    # Drive the robot
    motion_control(waypoint, robot_pose, desired_radius)
    
    # Update robot pose based on sensor feedback
    robot_pose = get_robot_pose()
    # Calculate the new distance to waypoint
    distance_to_waypoint = math.sqrt((waypoint[0] - robot_pose[0])**2 + (waypoint[1] - robot_pose[1])**2)
    # Check if the robot is within the desired radius of the waypoint
    if distance_to_waypoint <= desired_radius:
        #implement the delay of 2 seconds
        ppi.set_velocity([0, 0], turning_tick=0, time=2)
    elif distance_to_waypoint > desired_radius:
        #rerun the motion control function
        motion_control(waypoint, robot_pose, desired_radius)

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))


def get_robot_pose(args,script_dir):
    #################################################### --- Normal
    datadir,ip_ = args.calib_dir, args.ip
    camera_matrix = np.loadtxt("{}intrinsic.txt".format(datadir), delimiter=',')
    dist_coeffs = np.loadtxt("{}distCoeffs.txt".format(datadir), delimiter=',')
    scale = np.loadtxt("{}scale.txt".format(datadir), delimiter=',')
    if ip_ == 'localhost':
        scale /= 2
    baseline = np.loadtxt("{}baseline.txt".format(datadir) , delimiter=',')
    robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
    slam = EKF(robot)
    # update the robot pose [x,y,theta]
    robot_pose_act = slam.robot.state.tolist() # replace with your calculation

    #return robot_pose_act
    
    ###########################################################---Advanced part
    image_poses = {}
    with open(f'{script_dir}/lab_output/images.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['imgfname']] = pose_dict['pose']

    # estimate pose of targets in each image
    for image_path in image_poses.keys():
        input_image = cv2.imread(image_path)
        # cv2.imshow('bbox', bbox_img)
        # cv2.waitKey(0)
        robot_pose_cam = image_poses[image_path]
        
    weight_pose1 = 0.7
    weight_pose2 = 0.3
    
    print(f'robot_pose_cam: {robot_pose_cam}')
    print(f'robot_pose_act: {robot_pose_act}')
# Perform weighted fusion of poses
    robot_pose = [[],[],[]]
    
    robot_pose[0] = weight_pose1*robot_pose_act[0][0] + weight_pose2*robot_pose_cam[0][0]
    robot_pose[1] = weight_pose1*robot_pose_act[1][0] + weight_pose2*robot_pose_cam[1][0]
    robot_pose[2] = weight_pose1*robot_pose_act[2][0] + weight_pose2*robot_pose_cam[2][0]

    ###############################################################

    return robot_pose

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map_full.txt') # change to 'M4_true_map_part.txt' for lv2&3
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--port", metavar='', type=int, default=8080)
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip,args.port)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map) #list of fruits names, locations of fruits, locations of aruco markers
    search_list = read_search_list() #inputted ordered list of fruits to search 
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    waypoint = [0.0,0.0]

    # estimate the robot's pose
    robot_pose = get_robot_pose(args,os.path.dirname(os.path.abspath(__file__)))

    ##################################### New Week4 implementation

    start = np.array([0.0, 0.0])
    obstacles = fruits_true_pos.tolist() + aruco_true_pos.tolist()
    obs_radius = 1

    circle_obstacles = []
    for obs in obstacles:
        circle_obstacles.append(Circle(obs[0], obs[1], obs_radius))
        

    for i in range(len(fruits_true_pos)):
        goal = fruits_true_pos[i]
        rrtc = RRTC(start=start, goal=goal, width=16, height=10, obstacle_list=obstacles,
              expand_dis=3.0, path_resolution=1)

        path = rrtc.planning() 
        path = path.reverse()

        for point in path:
            drive_to_point(point, get_robot_pose(args,os.path.dirname(os.path.abspath(__file__))))        

        start = get_robot_pose(args,os.path.dirname(os.path.abspath(__file__)))
