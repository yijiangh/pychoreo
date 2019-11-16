import random

from pybullet_planning import multiply, interval_generator
from pybullet_planning import Pose, Point, Euler, INF

def extrusion_ee_pose_gen_fn(path_pts, orient_gen_fn, ee_pose_interp_fn, approach_distance=0.05, max_attempts=INF, landmark_only=False, **kwargs):
    assert len(path_pts) > 1, 'ee generation function must return at least two ee poses for interpolation.'
    attempt = 0
    for ee_orient in orient_gen_fn:
        if attempt > max_attempts: break
        extrude_from_approach = Pose(Point(z=-approach_distance))
        extrude_landmarks = [multiply(Pose(point=Point(*pt)), ee_orient) for pt in path_pts]
        ee_landmarks = []
        # approach
        ee_landmarks.append([multiply(extrude_landmarks[0], extrude_from_approach), extrude_landmarks[0]])
        # extrude
        ee_landmarks.append(extrude_landmarks)
        # retreat
        ee_landmarks.append([extrude_landmarks[-1], multiply(extrude_landmarks[-1], extrude_from_approach)])
        if landmark_only:
            yield ee_landmarks
        else:
            process_path = []
            for end_pts_path in ee_landmarks:
                sub_path = []
                for i in range(len(end_pts_path)-1):
                    sub_path.extend(ee_pose_interp_fn(end_pts_path[i], end_pts_path[i+1], **kwargs))
                process_path.append(sub_path)
            yield process_path
        attempt += 1
        # break
