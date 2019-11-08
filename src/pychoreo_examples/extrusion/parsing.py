from __future__ import print_function

import json
import os
import random
from collections import defaultdict
import numpy as np

from pybullet_planning import add_line, create_cylinder, set_point, Euler, quat_from_euler, \
    set_quat, get_movable_joints, set_joint_positions, pairwise_collision, Pose, multiply, Point, load_model, \
    HideOutput, load_pybullet, link_from_name, has_link, joint_from_name, apply_alpha, set_camera_pose
from pybullet_planning import RED

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
