from __future__ import print_function

import json
import os
import random
from collections import defaultdict
import numpy as np

from pybullet_planning import add_line, create_cylinder, set_point, Euler, quat_from_euler, \
    set_quat, get_movable_joints, set_joint_positions, pairwise_collision, Pose, multiply, Point, load_model, \
    HideOutput, load_pybullet, link_from_name, has_link, joint_from_name
# from choreo.assembly_datastructure import AssemblyElement

__all__ = [
    'load_extrusion',
    'create_elements',
]


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
    origin = parse_origin(json_data)
    return [origin + parse_point(json_node['point'], scale=scale) for json_node in json_data['node_list']]


def parse_ground_nodes(json_data):
    return {i for i, json_node in enumerate(json_data['node_list']) if json_node['is_grounded'] == 1}

##################################################

def draw_element(node_points, element, color=(1, 0, 0)):
    n1, n2 = element
    p1 = node_points[n1]
    p2 = node_points[n2]
    return add_line(p1, p2, color=color[:3])


def create_elements(node_points, elements, radius=0.001, color=(1, 0, 0, 1)):
    """ Create pybullet bodies for extrusion rods (cylinder)

    Parameters
    ----------
    node_points : type
        Description of parameter `node_points`.
    elements : type
        Description of parameter `elements`.
    radius : type
        Description of parameter `radius`.
    color : type
        Description of parameter `color`.

    Returns
    -------
    create_elements(node_points, elements, radius=0.0005,
        Description of returned object.

    """
    raise DeprecationWarning

    #radius = 0.0001
    #radius = 0.00005
    #radius = 0.000001
    radius = 1e-6
    # TODO: seems to be a min radius

    # TODO: just shrink the structure to prevent worrying about collisions at end-points
    # shrink = 0.01
    shrink = 0.005
    # shrink = 0.002
    #shrink = 0.
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

def get_node_neighbors(elements):
    node_neighbors = defaultdict(set)
    for e in elements:
        n1, n2 = e
        node_neighbors[n1].add(e)
        node_neighbors[n2].add(e)
    return node_neighbors


def get_element_neighbors(element_bodies):
    node_neighbors = get_node_neighbors(element_bodies)
    element_neighbors = defaultdict(set)
    for e in element_bodies:
        n1, n2 = e
        element_neighbors[e].update(node_neighbors[n1])
        element_neighbors[e].update(node_neighbors[n2])
        element_neighbors[e].remove(e)
    return element_neighbors
