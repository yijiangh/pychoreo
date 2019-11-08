import random
import numpy as np

from pybullet_planning import multiply, interval_generator
from pybullet_planning import Pose, Point, Euler


def get_random_direction_generator(**kwargs):
    lower = [-np.pi, -np.pi]
    upper = [+np.pi, +np.pi]
    for [roll, pitch] in interval_generator(lower, upper, **kwargs):
        pose = Pose(euler=Euler(roll=roll, pitch=pitch))
        yield pose


def get_enumeration_pose_generator(pose_list, shuffle=True):
    if shuffle:
        pose_list = random.choice(pose_list)
    for p in pose_list:
        yield p
