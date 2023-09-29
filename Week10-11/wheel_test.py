import numpy as np
# for computing the wheel calibration parameters
import numpy as np
import os
import sys
sys.path.insert(0, "util")
from pibot import PenguinPi

class WheelTest:
    def __init__(self, args):
        self.fileS = "{}scale.txt".format("calibration/param/")
        self.scale = np.loadtxt(self.fileS, delimiter=',')
        self.fileB = "{}baseline.txt".format("calibration/param/")  
        self.baseline = np.loadtxt(self.fileB, delimiter=',')

        self.drive_actual = []
        self.drive_setpoint = []

        self.turn_actual = []
        self.turn_setpoint = []

        self.ppi = PenguinPi(args.ip,args.port)

    def testDrive(self, ticks=20, dist=1):
        self.drive_setpoint.append(dist)
        dt = dist/(ticks*self.scale)
        print("driving for ", dt)
        self.ppi.set_velocity([1, 0], ticks, 0, dt)
        actual = float(input('Please enter the actual distance drove (m)'))
        self.drive_actual.append(actual)

    def testTurn(self, turning_ticks=5, theta=2*np.pi):
        self.turn_setpoint.append(theta)
        dt = (theta*self.baseline)/(2*self.scale*turning_ticks)
        print("turning for ", dt)
        self.ppi.set_velocity([0, 1], 0, turning_ticks, dt)
        actual = float(input('Please enter the actual distance drove (radians)'))
        self.turn_actual.append(actual)
    
    def summary(self):
        rmse = np.sqrt(np.mean([(i - j) ** 2 for (i, j) in zip(self.drive_actual,self.drive_setpoint)]))
        print("Driving RMSE is ", rmse)
        print("mean = ", np.mean(self.drive_actual))
        print("variance = ", np.var(self.drive_actual, ddof=1))

        turn_rmse = np.sqrt(np.mean([(i - j) ** 2 for (i, j) in zip(self.turn_actual,self.turn_setpoint)]))
        print("Turning RMSE is ", turn_rmse)
        print("turning mean = ", np.mean(self.turn_actual))
        print("turning variance = ", np.var(self.turn_actual, ddof=1))


def ask_yesno(question):
    ans = ""
    while True:
        ans = input(question + " [y/n] ")
        if ans == "y":
            return True
        elif ans == "n":
            return False 

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    args, _ = parser.parse_known_args()
    
    w = WheelTest(args)

    while(True):
        if ask_yesno("Do you want to test driving the robot forward?"):
            w.testDrive()
        else:
            break

    while(True):
        if ask_yesno("Do you want to test driving turning the robot?"):
            w.testTurn()
        else:
            break
    
    w.summary()