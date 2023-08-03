# for taking a photo of the calibration rig
import numpy as np
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, "../util")
from util.pibot import PenguinPi
import pygame

class calibration:
    def __init__(self,args):
        self.pibot = PenguinPi(args.ip, args.port)
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.command = {'motion':[0, 0], 'image': False}
        self.finish = False

    def image_collection(self, dataDir, images_to_collect):
        if self.command['image']:
            for i in range(images_to_collect):
                image = self.pibot.get_image()
                filename = "calib_{}.png".format(i)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename, image)
            self.finish = True

    def update_keyboard(self):
        for event in pygame.event.get():
            ########### replace with your M1 codes ###########
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                pass # TODO: replace with your M1 code to make the robot drive forward
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                pass # TODO: replace with your M1 code to make the robot drive backward
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                pass # TODO: replace with your M1 code to make the robot turn left
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                pass # TODO: replace with your M1 code to make the robot turn right
            ####################################################
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                self.command['image'] = True

    def control(self):
        motion_command = self.command['motion']
        lv, rv = self.pibot.set_velocity(motion_command)

    def take_pic(self):
        self.img = self.pibot.get_image()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    args, _ = parser.parse_known_args()

    currentDir = os.getcwd()
    dataDir = "{}/param/".format(currentDir)
    if not os.path.exists(dataDir):
        os.makedirs(dataDir)
    
    images_to_collect = 1

    calib = calibration(args)

    width, height = 640, 480
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Calibration')
    canvas.fill((0, 0, 0))
    pygame.display.update()
    
    # collect data
    print('Collecting {} images for camera calibration.'.format(images_to_collect))
    print('Press ENTER to capture image.')
    while not calib.finish:
        
        calib.update_keyboard()
        calib.control()
        calib.take_pic()
        calib.image_collection(dataDir, images_to_collect)
        img_surface = pygame.surfarray.make_surface(calib.img)
        img_surface = pygame.transform.flip(img_surface, True, False)
        img_surface = pygame.transform.rotozoom(img_surface, 90, 1)
        canvas.blit(img_surface, (0, 0))
        pygame.display.update()
    print('Finished image collection.\n')


