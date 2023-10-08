import os
import sys
import time
import cv2
import numpy as np
import math
import json

from rrt3 import RrtConnect
#from rrt import RRTC
from Obstacle import *
from operate import Operate
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


def drive_to_waypoint(waypoint, robot_pose):
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

    target_angle = turn_to_waypoint(waypoint, robot_pose)

    # Drive straight to waypoint
    distance_to_waypoint = math.sqrt((waypoint_x - robot_pose_x)**2 + (waypoint_y - robot_pose_y)**2)
    drive_time = abs(float((distance_to_waypoint) / (wheel_ticks*scale))) 

    print("Driving for {:.2f} seconds".format(drive_time))
    motion_controller([1,0], wheel_ticks, drive_time)
    print("Arrived at [{}, {}]".format(waypoint[1], waypoint[0]))
    
    update_robot_pose = [waypoint[0], waypoint[1], target_angle]
    return update_robot_pose 

def turn_to_waypoint(waypoint, robot_pose):
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
        motion_controller([0,1], wheel_ticks, turn_time)
    elif theta_delta < 0:
        motion_controller([0,-1], wheel_ticks, turn_time)
    else:
        print("There is an issue with turning function")
        
    return theta_delta # delete once we get the robot_pose working and path plannning
    
    
    
def motion_controller(motion, wheel_ticks, drive_time):
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
            
def coords_to_obstacles(fruits_true_pos, aruco_true_pos):  
    obstacles = []
    
    for x,y in fruits_true_pos:
        # Create a circle object for all fruits
        circle = Circle(x, y, 0.1)
        obstacles.append(circle)
        
    for x,y in aruco_true_pos:
        # Create square object for all aruco markers
        square = Rectangle(np.array([x,y], 0.1, 0.1))
        obstacles.append(square)   
    
    return obstacles   
    
    

# main loop
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--map", type=str, default='M4_true_map_full.txt')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip,args.port)
    operate = Operate(args)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map) #list of fruits names, locations of fruits, locations of aruco markers
    search_list = read_search_list() #inputted ordered list of fruits to search 
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    print(f'fruits_true_pos: {fruits_true_pos}')
    print(f'search list: {search_list}')
    
    # Define starting and robot pose 
    start = np.array([0.0, 0.0])
    robot_pose =  np.array([0.0, 0.0, 0.0])
    waypoint = [0.0,0.0]
    
    ############Obstacles######################    
    start = np.array([0.0, 0.0])
    #obstacles = fruits_true_pos.tolist() + aruco_true_pos.tolist() # MAKE SURE ACURO IS SQURE AND FRUIT IS CIRCLE!!!!!
    #obs_radius = 0.1
    
    obstacles = coords_to_obstacles(fruits_true_pos, aruco_true_pos)
    print("Obstacle fruit is:", obstacles)
    ###########################################
    
    
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


    ####################################
    # Within a while loop going through the search list create a path using rrt
    
    
    # Then run the robot to go to way point by iterating through the waypoint list
    
    
    ###########################################

    
    ############Path Planning######################
    print("Entering fruits loop")
    path_list = []
    for i in range(len(fruits_true_pos)):
        goal = fruits_true_pos[i]
        
        waypoint_x, waypoint_y = input("Enter waypoint (x,y): ").split(",")
        waypoint = [float(waypoint_y), float(waypoint_x)]
        
        robot_pose = drive_to_waypoint(waypoint, robot_pose, args)
        #robot_pose[2] = math.atan2((waypoint[1] - robot_pose[1]), (waypoint[0] - robot_pose[0]))
        #robot_pose[0:2] = waypoint

sys.exit()
    