import random

from pybullet_planning import multiply, interval_generator
from pybullet_planning import Pose, Point, Euler, INF

def picknplace_ee_pose_gen_fn(unit_geo, ee_pose_interp_fn, landmark_only=False, **kwargs):
    for initial_frame, pick_grasp, goal_frame, place_grasp in zip(
                                         unit_geo.get_initial_frames(get_pb_pose=True), unit_geo.pick_grasps,\
                                         unit_geo.get_goal_frames(get_pb_pose=True),    unit_geo.place_grasps):
        pick_approach_landmarks = [multiply(initial_frame, pick_grasp.get_object_from_approach_frame(get_pb_pose=True)), \
                                   multiply(initial_frame, pick_grasp.get_object_from_attach_frame(get_pb_pose=True))]
        pick_retreat_landmarks = [multiply(initial_frame, pick_grasp.get_object_from_attach_frame(get_pb_pose=True)), \
                                  multiply(initial_frame, pick_grasp.get_object_from_retreat_frame(get_pb_pose=True))]

        place_approach_landmarks = [multiply(goal_frame, place_grasp.get_object_from_approach_frame(get_pb_pose=True)), \
                                    multiply(goal_frame, place_grasp.get_object_from_attach_frame(get_pb_pose=True))]
        place_retreat_landmarks = [multiply(goal_frame, place_grasp.get_object_from_attach_frame(get_pb_pose=True)), \
                                   multiply(goal_frame, place_grasp.get_object_from_retreat_frame(get_pb_pose=True))]

        ee_landmarks = [pick_approach_landmarks, pick_retreat_landmarks, place_approach_landmarks, place_retreat_landmarks]

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
