import random

from pybullet_planning import multiply, interval_generator
from pybullet_planning import Pose, Point, Euler, INF

def extrusion_ee_pose_gen_fn(path_pts, direction_gen_fn, ee_pose_interp_fn, max_attempts=INF, landmark_only=False, **kwargs):
    assert len(path_pts) > 1, 'ee generation function must return at least two ee poses for interpolation.'
    attempt = 0
    while attempt < max_attempts:
        ee_orient = next(direction_gen_fn)
        ee_landmarks = [multiply(Pose(point=Point(*pt)), ee_orient) for pt in path_pts]
        if landmark_only:
            yield ee_landmarks
        else:
            full_path = []
            for i in range(len(ee_landmarks)-1):
                full_path.extend(ee_pose_interp_fn(ee_landmarks[i], ee_landmarks[i+1], **kwargs))
            yield full_path
        attempt += 1
        # break
