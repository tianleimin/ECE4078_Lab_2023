import os
import sys
import time
import cv2
import numpy as np
import math
import json

#from rrt3 import RrtConnect
#from rrt import RRTC
from Obstacle import *
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


def drive_to_waypoint(waypoint, robot_pose, args):
    datadir,_ = args.calib_dir, args.ip

    # Read in baseline and scale parameters
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


    # Calculate the angle from robot's current position to the target waypoint
    target_angle = math.atan2(waypoint_y - robot_pose_y, waypoint_x - robot_pose_x)

    # Calculate the angle to turn to align with the target angle
    angle_to_turn = target_angle - robot_pose_theta


    '''
    # Ensure the angle is within the range of -pi to pi
    if angle_to_turn > math.pi:
        angle_to_turn -= 2 * math.pi
    elif angle_to_turn < -math.pi:
        angle_to_turn += 2 * math.pi
    '''
    if target_angle < 0:
        angle = angle_to_turn - 2*np.pi
    else:
        angle = angle_to_turn + 2*np.pi
        
    if abs(angle_to_turn) > abs(angle):
        theta_error = angle
    else:
        theta_error = angle_to_turn
        
    print(f'Angle to turn {theta_error}')
    
    # Determine turn time
    turn_time = abs(float(angle_to_turn / (2 *np.pi * baseline)))
    #turn_time = float((abs(angle_to_turn)*baseline)/(2*wheel_ticks*scale))
    print("Turning for {:.2f} seconds".format(turn_time))

    if theta_error > 0:
         # Rotate counterclockwise (left)
        ppi.set_velocity([0, -1], turning_tick=wheel_ticks, time=turn_time)
    else:
        # Rotate clockwise (right)
        ppi.set_velocity([0, 1], turning_tick=wheel_ticks, time=turn_time)
    '''
    # Determine the direction of rotation (clockwise or counterclockwise)
    if angle_to_turn > 0:
        rotation_direction = "clockwise"
    else:
        rotation_direction = "counterclockwise"

    # Set the velocity and turning direction based on rotation_direction
    if rotation_direction == "clockwise":
        # Rotate clockwise (right)
        ppi.set_velocity([0, 1], turning_tick=wheel_ticks, time=turn_time)
    elif rotation_direction == "counterclockwise":
        # Rotate counterclockwise (left)
        ppi.set_velocity([0, -1], turning_tick=wheel_ticks, time=turn_time)
    else:
        # No rotation needed
        ppi.set_velocity([0, 0], turning_tick=wheel_ticks, time=0)
    '''

    # Drive straight to waypoint
    distance_to_waypoint = math.sqrt((waypoint_x - robot_pose_x)**2 + (waypoint_y - robot_pose_y)**2)
    drive_time = abs(float((distance_to_waypoint) / (wheel_ticks*scale))) 

    print("Driving for {:.2f} seconds".format(drive_time))
    ppi.set_velocity([1, 0], tick=wheel_ticks, time=drive_time)
    
    print("Arrived at [{}, {}]".format(waypoint[1], waypoint[0]))
    
    update_robot_pose = [waypoint[0], waypoint[1], target_angle]
    return update_robot_pose 



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

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map) #list of fruits names, locations of fruits, locations of aruco markers
    search_list = read_search_list() #inputted ordered list of fruits to search 
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    print(f'fruits_true_pos: {fruits_true_pos}')
    print(f'search list: {search_list}')
    
    # Define starting and robot pose 
    start = np.array([0.0, 0.0])
    robot_pose =  np.array([0.0, 0.0, 0.0])
    #waypoint = [0.0,0.0]
    
    ############Obstacles######################    
    start = np.array([0.0, 0.0])
    obstacles = fruits_true_pos.tolist() + aruco_true_pos.tolist() # MAKE SURE ACURO IS SQURE AND FRUIT IS CIRCLE!!!!!
    obs_radius = 0.1
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
    