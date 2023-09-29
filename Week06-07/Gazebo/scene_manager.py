#!/usr/bin/env python
import sys
import math
import os
import random
import json
from math import pi

import copy
import rospy
import tf
from gazebo_msgs.srv import DeleteModel, SpawnModel
from geometry_msgs.msg import *
from geometry_msgs.msg import Point, Quaternion
import numpy as np

import rospkg

rospack = rospkg.RosPack()

class SceneManager:
    def __init__(self):
        self.obj_class_dict = {
				'aruco1':1, 'aruco2':1, 'aruco3':1, 'aruco4':1, 'aruco5':1, 
				'aruco6':1, 'aruco7':1, 'aruco8':1, 'aruco9':1, 'aruco10':1,
				'apple':2, 'lemon':2, 'strawberry':2, 'orange':2, 'pear':2}
        # print(self.obj_class_dict)
        rospack = rospkg.RosPack()

        self.workspace_path =  rospack.get_path('penguinpi_gazebo')
        urdf_lib_path = self.workspace_path + '/models/aruco/urdf/'
        self.urdf_dict = {}
        for key in self.obj_class_dict:
            urdf_file_name = urdf_lib_path + key + '.urdf'
            with open(urdf_file_name) as f:
                self.urdf_dict[key] = f.read()
        print(self.urdf_dict.keys())

    def spawn_all_objs(self, save_map_to):
        print('Waiting for gazebo services...')
        rospy.wait_for_service('gazebo/spawn_urdf_model')
        spawn_model = rospy.ServiceProxy('gazebo/spawn_urdf_model', SpawnModel)
        nx, ny = (7, 7)
        x = np.linspace(-1.2, 1.2, nx)
        y = np.linspace(-1.2, 1.2, ny)
        xv, yv = np.meshgrid(x, y)
        rand_idx = np.arange(nx*ny)
        random.shuffle(rand_idx)
        random.shuffle(rand_idx)
        obj_counter = 0
        map_dict = {}
        for key in self.obj_class_dict.keys():
            for i in range(0, int(self.obj_class_dict[key])):
                x_idx = int(rand_idx[obj_counter]//nx)
                y_idx = int(rand_idx[obj_counter]%ny)
                x_temp = xv[x_idx, y_idx]
                y_temp = yv[x_idx, y_idx]
                obj_counter += 1
                item_name = '%s_%i' % (key, i)
                # print('Spawning model:%s', item_name)
                quat = np.array(
                    tf.transformations.quaternion_from_euler(0, 0, 0))
                target_pose = Pose(Point(x_temp, y_temp, 0.06),
                                   Quaternion(quat[0],
                                              quat[1], quat[2], quat[3]))
                spawn_model(item_name, self.urdf_dict[key], '',
                            target_pose, 'world')
                map_dict[item_name] = {'x': x_temp, 'y': y_temp}
        if save_map_to != '':
            f_path = self.workspace_path + '/' + save_map_to
            print(f_path)
            with open(f_path, 'w') as f:
                json.dump(map_dict, f)

    def spawn_from_file(self, map_file_name):
        print('Waiting for gazebo services...')
        rospy.wait_for_service('gazebo/spawn_urdf_model')
        spawn_model = rospy.ServiceProxy('gazebo/spawn_urdf_model', SpawnModel)
        path = self.workspace_path + '/'+ map_file_name
        try:
            map = json.load(open(path))
            for item_name, loc in map.items():
                x_temp = loc['x']
                y_temp = loc['y']
                urdf_key = item_name.split('_')[0]
                quat = np.array(
                    tf.transformations.quaternion_from_euler(0, 0, 0))
                target_pose = Pose(Point(x_temp, y_temp, 0.06),
                                    Quaternion(quat[0],
                                                quat[1], quat[2], quat[3]))
                spawn_model(item_name, self.urdf_dict[urdf_key], '',
                            target_pose, 'world')
        except IOError:
            print('path %s does not exist' % path)

    def delete_all_objs(self):
        for key in self.obj_class_dict:
            for i in range(0, int(self.obj_class_dict[key])):
                obj_name = '%s_%i' % (key, i)
                # print('deleting model:%s', obj_name)
                self.delete_model(obj_name)
                rospy.sleep(0.05)

    def delete_model(self, model_name):
        try:
            delete_model = rospy.ServiceProxy('gazebo/delete_model',
                                              DeleteModel)
            delete_model(model_name)
        except rospy.ServiceException as e:
            print 'Service call failed: %s'


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--delete_all', action='store_true')
    parser.add_argument('-l', '--load_map_from', type=str, default='')
    parser.add_argument('-s', '--save_map_to', type=str, default='')
    args, _ = parser.parse_known_args()

    rospy.init_node('spawn_objects')
    obj_manager = SceneManager()
    if args.delete_all:
        obj_manager.delete_all_objs()
    elif args.load_map_from == '':
        obj_manager.spawn_all_objs(args.save_map_to)
    else:
        obj_manager.spawn_from_file(args.load_map_from)
