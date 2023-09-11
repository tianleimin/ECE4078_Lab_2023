# access each wheel and the camera onboard of PenguinPi

import numpy as np
import requests
import cv2 
import time
import urllib.request


class PenguinPi:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.wheel_vel = [0, 0]

    ##########################################
    # Change the robot velocity here
    # tick = forward speed
    # turning_tick = turning speed
    ########################################## 
    def set_velocity(self, command, tick=50, turning_tick=20, time=0):
        # running in sim
        if self.ip == 'localhost': 
            l_vel = command[0]*tick - command[1]*turning_tick
            r_vel = command[0]*tick + command[1]*turning_tick
        # running on physical robot (if right wheel spins backwards in test motor)
        else:
            l_vel = command[0]*tick - command[1]*turning_tick
            r_vel = -command[0]*tick - command[1]*turning_tick # reverse right wheel velocity
        self.wheel_vel = [l_vel, r_vel]
        if time == 0:
            requests.get(
                f"http://{self.ip}:{self.port}/robot/set/velocity?value="+str(l_vel)+","+str(r_vel))
        else:
            assert (time > 0), "Time must be positive."
            assert (time < 30), "Time must be less than network timeout (20s)."
            requests.get(
                "http://"+self.ip+":"+str(self.port)+"/robot/set/velocity?value="+str(l_vel)+","+str(r_vel)
                            +"&time="+str(time))
        return l_vel, r_vel
    
    # get frame from simulated robot    
    def get_image_sim(self):
        try:
            r = requests.get(f"http://{self.ip}:{self.port}/camera/get", timeout=0.1)
            img = cv2.imdecode(np.frombuffer(r.content,np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            print("Image retrieval timed out.")
            img = np.zeros((240,320,3), dtype=np.uint8)
        return img
    
    # get frames from the physical robot
    def get_image_physical(self):
        try:
            # the image stream URL
            url_str = f"http://{self.ip}:{self.port}/camera/get" # "http://192.168.50.1:8080/camera/get"
            #encoding = 'ISO-8859-1'
            x = urllib.request.urlopen(url=url_str)

            # size of bytes to read from the connected robot for a frame
            max_size = 2048
            result = b''
            # look for two consecutive '--frame' in the data
            i = 0
            while True:
                buf = x.fp.read(max_size)
                # (Second) frame start?
                if b'--frame' in buf:
                    i += 1
                result += buf
                if i > 1:
                    break
            next_frame_boundary = result.rfind(b'--frame')
            # get the binary data in between '--frame'
            img_bits = result[len(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'):next_frame_boundary]
            # save the binary data as image for display
            img_array = np.frombuffer(img_bits, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            print("Image retrieval timed out.")
            img = np.zeros((240,320,3), dtype=np.uint8)
        return img

    # choose which get image function to use depending on running in simulator or on robot
    def get_image(self):
        if self.ip == 'localhost':
            return self.get_image_sim()
        else:
            return self.get_image_physical()
