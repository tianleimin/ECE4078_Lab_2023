"""
Please note that this test is not a definitive answer of right or wrong implementation
it simply evaluate if the required TODO is what we expected from the skeleton code we 
provided. There is no point in making your code pass this test and obtain worse localization
result!
"""

from operate import Operate
import pytest
import logging
import numpy as np
from util.measure import *
import pickle


class fake_args:
    def __init__(self):
        self.play_data = False
        self.save_data = False
        self.calib_dir = "calibration/param/"
        self.ip = "127.0.0.1"
        self.port = "40000"
logger = logging.getLogger(__name__)


args = fake_args()   
with open("test/slam_test.pk", "rb") as pk_file:
    expected_data = pickle.load(pk_file)




test_drive_measurement = [
    Drive(0, 0, 0.1, left_cov = 1, right_cov = 1),
    Drive(1.0, 1.0, 0.1, left_cov = 1, right_cov = 1),
    Drive(2.0, 1.0, 0.1, left_cov = 1, right_cov = 1),
    Drive(0, 0, 0.1, left_cov = 1, right_cov = 1)
]

def test_wheel_scale_calibration():
    operate = Operate(args)
    robot = operate.ekf.robot
    if robot.wheels_scale == 2.635203168862666930e-03:
        pytest.skip("You haven't calibrated the wheel_scale")

def test_wheel_width_calibration():
    operate = Operate(args)
    robot = operate.ekf.robot
    if robot.wheels_width == 1.535024533975867245e-01:
        pytest.skip("You haven't calibrated the wheel_width")

def test_intrinsic_calibration():
    operate = Operate(args)
    robot = operate.ekf.robot
    default_intrinsic = [3.101321409128037772e+02,0.000000000000000000e+00,1.564094633411971529e+02,
                        0.000000000000000000e+00,3.109819011913432405e+02,1.227391164086467768e+02,
                        0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]
    if np.all(np.isclose(robot.camera_matrix.flatten(), np.array(default_intrinsic))):
        pytest.skip("You haven't calibrated the camera")

def test_derivative_drive():
    operate = Operate(args)
    robot = operate.ekf.robot
    for i, drive_meas in enumerate(test_drive_measurement):
        DFx = robot.derivative_drive(drive_meas)
        assert np.all(np.isclose(DFx, expected_data["DFx"][i])), "wrong implementation of derivative_drive"

def test_covariance_drive():
    operate = Operate(args)
    robot = operate.ekf.robot
    for i, drive_meas in enumerate(test_drive_measurement):
        cov = robot.covariance_drive(drive_meas)
        assert np.all(np.isclose(cov, expected_data["cov"][i])), "wrong implementation of covariance_drive"

def test_ekf_predict():
    operate = Operate(args)
    robot = operate.ekf.robot
    ekf = operate.ekf
    for i, drive_meas in enumerate(test_drive_measurement):
        ekf.predict(drive_meas)
        assert np.all(np.isclose(ekf.P, expected_data["P"][i])), "wrong calculation on P"
        assert np.all(np.isclose(robot.state, expected_data["predict_state"][i])), "You have not advanced the kinematics model"

def test_ekf_update():
    operate = Operate(args)
    robot = operate.ekf.robot
    ekf = operate.ekf

    expected_state = expected_data["slam_data"]["expected_state"]
    controls = expected_data["slam_data"]["control"]
    measurements = expected_data["slam_data"]["measurement"]

    states = []
    for i, img_meas in enumerate(measurements):
        drive_signal = Drive(controls[i, 0], controls[i, 1], dt = 0.2)
        ekf.predict(drive_signal)
        ekf.add_landmarks(img_meas)
        ekf.update(img_meas)
        states.append(robot.state)
    states = np.array(states).squeeze(2)

    assert np.all(np.isclose(states, expected_state)), "MAYBE wrong implementation of update function"