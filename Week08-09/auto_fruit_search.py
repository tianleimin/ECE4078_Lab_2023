# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time

# import SLAM components
# sys.path.insert(0, "{}/slam".format(os.getcwd()))
# from slam.ekf import EKF
# from slam.robot import Robot
# import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
from pibot import PenguinPi
import measure as measure


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
        for i in range(3):
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
import math
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
    
    # maybe just everything that controls the motion of the robot within a function so i can rerun it
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
    drive_time = (distance_to_waypoint-radius) / linear_velocity # could minus 0.5m from waypoint to get to radius of 0.5

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


def get_robot_pose():
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    # update the robot pose [x,y,theta]
    robot_pose = [0.0,0.0,0.0] # replace with your calculation
    ####################################################

    return robot_pose


class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


#def astar(maze, start, end):
def astar(search_list, fruits_list, fruits_true_pos, aruco_true_pos):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    #max dist. from (0,0)
    boundary = 1.5 #metres

    #define start and end targets
    start = (0,0)
    end_target = search_list[-1]
    for target in fruits_true_pos:
        if target[0] == end_target:
            end = target[1]
            break

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []

       
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if abs(node_position[0]) > boundary or abs(node_position[1]) > boundary:
                continue

            # Make sure walkable terrain
            #check if node_position is not an aruco marker or fruit
            for fruit in fruits_true_pos:
                if fruit == node_position:
                    continue

            for marker in aruco_true_pos:
                if marker == node_position:
                    continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)


# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map_full.txt') # change to 'M4_true_map_part.txt' for lv2&3
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip,args.port)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]

    path_planner()

    # The following is only a skeleton code for semi-auto navigation
    while True:
        # enter the waypoints
        # instead of manually enter waypoints, you can give coordinates by clicking on a map, see camera_calibration.py from M2
        x,y = 0.0,0.0
        x = input("X coordinate of the waypoint: ")
        try:
            x = float(x)
        except ValueError:
            print("Please enter a number.")
            continue
        y = input("Y coordinate of the waypoint: ")
        try:
            y = float(y)
        except ValueError:
            print("Please enter a number.")
            continue

        # estimate the robot's pose
        robot_pose = get_robot_pose()

        # robot drives to the waypoint
        waypoint = [x,y]
        drive_to_point(waypoint,robot_pose)
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

        # exit
        ppi.set_velocity([0, 0])
        uInput = input("Add a new waypoint? [Y/N]")
        if uInput == 'N':
            break
