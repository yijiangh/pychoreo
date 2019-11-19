from __future__ import print_function

import json
import os
import random
import datetime
from collections import defaultdict, OrderedDict
import numpy as np

from pybullet_planning import add_line, create_cylinder, set_point, Euler, quat_from_euler, \
    set_quat, get_movable_joints, set_joint_positions, pairwise_collision, Pose, multiply, Point, load_model, \
    HideOutput, load_pybullet, link_from_name, has_link, joint_from_name, apply_alpha, set_camera_pose, is_connected
from pybullet_planning import RED

from pychoreo.process_model.trajectory import MotionTrajectory
from pychoreo_examples.extrusion.trajectory import PrintTrajectory, PrintBufferTrajectory

LENGTH_SCALE_CONVERSION = {
    'millimeter': 1e-3,
    'meter': 1.0,
}

##################################################

def load_extrusion(file_path, parse_layers=False, verbose=False):
    with open(file_path, 'r') as f:
        json_data = json.loads(f.read())
    scale = LENGTH_SCALE_CONVERSION[json_data['unit']]
    elements = parse_elements(json_data, parse_layers)
    node_points = parse_node_points(json_data, scale)
    ground_nodes = parse_ground_nodes(json_data)
    if verbose:
        print('Assembly: {} | Model: {} | Unit: {}'.format(
            json_data['assembly_type'], json_data['model_type'], json_data['unit'])) # extrusion, spatial_frame, millimeter
        print('Nodes: {} | Ground: {} | Elements: {}'.format(
            len(node_points), len(ground_nodes), len(elements)))
    return elements, node_points, ground_nodes


def parse_point(json_point, scale=1.0):
    return scale * np.array([json_point['X'], json_point['Y'], json_point['Z']])


def parse_transform(json_transform, scale=1.0):
    transform = np.eye(4)
    transform[:3, 3] = parse_point(json_transform['Origin'], scale=scale) # Normal
    transform[:3, :3] = np.vstack([parse_point(json_transform[axis], scale=1)
                                   for axis in ['XAxis', 'YAxis', 'ZAxis']])
    return transform


def parse_origin(json_data, scale=1.0):
    return parse_point(json_data['base_frame_in_rob_base']['Origin'], scale)


def parse_elements(json_data, parse_layers=False):
    # if not parse_layers:
    return [tuple(json_element['end_node_ids'])
        for json_element in json_data['element_list']]
    # else:
    #     return [AssemblyElement(node_ids=json_element['end_node_ids'], layer_id=json_element['layer_id'], e_id=i)
    #             for i, json_element in enumerate(json_data['element_list'])]


def parse_node_points(json_data, scale=1.0):
    origin = parse_origin(json_data, scale)
    return [origin + parse_point(json_node['point'], scale=scale) for json_node in json_data['node_list']]


def parse_ground_nodes(json_data):
    return {i for i, json_node in enumerate(json_data['node_list']) if json_node['is_grounded'] == 1}

##################################################

def create_elements_bodies(node_points, elements, radius=0.0015, shrink=0.002, color=apply_alpha(RED, alpha=1)):
    # TODO: just shrink the structure to prevent worrying about collisions at end-points
    # TODO: could scale the whole environment
    # radius = 1e-6 # 5e-5 | 1e-4
    # TODO: seems to be a min radius
    # 0. | 0.002 | 0.005

    element_bodies = []
    for (n1, n2) in elements:
        p1, p2 = node_points[n1], node_points[n2]
        height = max(np.linalg.norm(p2 - p1) - 2*shrink, 0)
        #if height == 0: # Cannot keep this here
        #    continue
        center = (p1 + p2) / 2
        # extents = (p2 - p1) / 2
        body = create_cylinder(radius, height, color=color)
        set_point(body, center)
        element_bodies.append(body)

        delta = p2 - p1
        x, y, z = delta
        phi = np.math.atan2(y, x)
        theta = np.math.acos(z / np.linalg.norm(delta))
        set_quat(body, quat_from_euler(Euler(pitch=theta, yaw=phi)))
        # p1 is z=-height/2, p2 is z=+height/2
    return element_bodies

##################################################

def export_trajectory(save_dir, trajs, ee_link_name, overwrite=True, shape_file_path='', indent=None, include_robot_data=True, include_link_path=True):
    if include_robot_data and include_link_path:
        assert is_connected(), 'needs to be connected to a pybullet client to get robot/FK data'

    if os.path.exists(shape_file_path):
        with open(shape_file_path, 'r') as f:
            shape_data = json.loads(f.read())
        if 'model_name' in shape_data:
            file_name = shape_data['model_name']
        else:
            file_name = shape_file_path.split('.json')[-2].split(os.sep)[-1]
    else:
        file_name = 'pychoreo_result'
        overwrite = False

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = OrderedDict()
    data['assembly_type'] = 'extrusion'
    data['file_name'] = file_name
    data['write_time'] = str(datetime.datetime.now())

    data['trajectory'] = []
    for cp_id, cp_trajs in enumerate(trajs):
        for sp_traj in cp_trajs:
            ee_link_path = sp_traj.get_link_path(ee_link_name)
        data['trajectory'].append([sp_traj.to_data(include_robot_data=True, include_link_path=True) for sp_traj in cp_trajs])

    full_save_path = os.path.join(save_dir, '{}_result_{}.json'.format(file_name,  '_'+data['write_time'] if not overwrite else ''))
    with open(full_save_path, 'w') as f:
        json.dump(data, f, indent=indent)

def parse_saved_trajectory(file_path):
    with open(file_path, 'r')  as f:
        data = json.load(f)
    print('file name: {} | write_time: {} | '.format(data['file_name'], data['write_time']))
    full_traj = []
    for proc_traj_data in data['trajectory']:
        proc_traj_recon = []
        assert proc_traj_data[0]['traj_type'] == 'MotionTrajectory'
        proc_traj_recon.append(MotionTrajectory.from_data(proc_traj_data[0]))

        assert proc_traj_data[1]['traj_type'] == 'PrintBufferTrajectory' and proc_traj_data[1]['tag'] == 'approach'
        proc_traj_recon.append(PrintBufferTrajectory.from_data(proc_traj_data[1]))

        assert proc_traj_data[2]['traj_type'] == 'PrintTrajectory'
        proc_traj_recon.append(PrintTrajectory.from_data(proc_traj_data[2]))

        assert proc_traj_data[3]['traj_type'] == 'PrintBufferTrajectory' and proc_traj_data[3]['tag'] == 'retreat'
        proc_traj_recon.append(PrintBufferTrajectory.from_data(proc_traj_data[3]))
        full_traj.append(proc_traj_recon)
    return full_traj


