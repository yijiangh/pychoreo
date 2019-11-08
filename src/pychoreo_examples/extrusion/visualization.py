import time
import numpy as np
from pybullet_planning import set_camera_pose, add_line, add_text, wait_for_user, LockRenderer
from pybullet_planning import unit_pose, multiply, tform_point
from pybullet_planning import Pose, Point, Euler, INF

##################################################

def draw_element(node_points, element, color=(1, 0, 0)):
    n1, n2 = element
    p1 = node_points[n1]
    p2 = node_points[n2]
    return add_line(p1, p2, color=color[:3])


##################################################

def set_extrusion_camera(node_points):
    centroid = np.average(node_points, axis=0)
    camera_offset = 0.25 * np.array([1, -1, 1])
    set_camera_pose(camera_point=centroid + camera_offset, target_point=centroid)

##################################################

def draw_extrusion_sequence(node_points, element_bodies, element_sequence, seq_poses=None, ee_pose_map_fn=lambda id : unit_pose,
                           line_width=10, time_step=INF, direction_len=0.005):
    assert len(element_bodies) == len(element_sequence)
    if seq_poses:
        assert len(seq_poses) == len(element_sequence)
    handles = []

    for seq_id, element in enumerate(element_sequence):
        n1, n2 = element
        p1, p2 = (node_points[n1], node_points[n2])
        e_mid = (np.array(p1) + np.array(p2)) / 2

        seq_ratio = float(seq_id)/(len(element_sequence)-1)
        color = np.array([0, 0, 1])*(1-seq_ratio) + np.array([1,0,0])*seq_ratio
        handles.append(add_line(p1, p2, color=tuple(color), width=line_width))
        handles.append(add_text(str(seq_id), position=e_mid))

        if seq_poses is not None:
            with LockRenderer():
                for flag_id, feasible_flag in enumerate(seq_poses[element]):
                    if feasible_flag:
                        ee_pose = ee_pose_map_fn(flag_id)
                        fmap_pose = multiply(Pose(point=e_mid), ee_pose)
                        origin_world = tform_point(fmap_pose, np.zeros(3))
                        axis = np.zeros(3)
                        axis[2] = 1
                        axis_world = tform_point(fmap_pose, direction_len*axis)
                        handles.append(add_line(origin_world, axis_world, color=axis))
                        # handles.append(draw_pose(cmap_pose, direction_len))
        if time_step < INF:
            time.sleep(time_step)
        else:
            wait_for_user()
