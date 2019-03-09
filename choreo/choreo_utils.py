import os
import random
import time
import numpy as np
import pybullet as p
from itertools import product
import json
from collections import OrderedDict, namedtuple

from conrob_pybullet.ss_pybullet.pybullet_tools.utils import add_line, Euler, \
    set_joint_positions, pairwise_collision, Pose, multiply, Point, HideOutput, load_pybullet, link_from_name, \
    get_link_pose, invert, get_bodies, set_pose, add_text, CLIENT, BASE_LINK, get_self_link_pairs, get_custom_limits, all_between, pairwise_link_collision, \
    tform_point
from conrob_pybullet.utils.ikfast.kuka_kr6_r900.ik import sample_tool_ik

END_EFFECTOR_PATH = '../conrob_pybullet/models/kuka_kr6_r900/urdf/extrusion_end_effector.urdf'
EE_TOOL_BASE_LINK = 'eef_base_link'
EE_TOOL_TIP = 'eef_tcp_frame'

# end effector direction discretization
THETA_DISC = 10
PHI_DISC = 20

WAYPOINT_DISC_LEN = 0.005 # meter
KINEMATICS_CHECK_TIMEOUT = 2

class MotionTrajectory(object):
    def __init__(self, robot, joints, path, attachments=[]):
        self.robot = robot
        self.joints = joints
        self.path = path
        self.attachments = attachments
    def reverse(self):
        return self.__class__(self.robot, self.joints, self.path[::-1], self.attachments)
    def iterate(self):
        for conf in self.path[1:]:
            set_joint_positions(self.robot, self.joints, conf)
            yield
    def __repr__(self):
        return 'm({},{})'.format(len(self.joints), len(self.path))

class PrintTrajectory(object):
    def __init__(self, robot, joints, path, element, reverse, colliding=set()):
        self.robot = robot
        self.joints = joints
        self.path = path
        self.n1, self.n2 = reversed(element) if reverse else element
        self.element = element
        self.colliding = colliding
    def __repr__(self):
        return '{}->{}'.format(self.n1, self.n2)

###########################################

def add_text(text, position=(0, 0, 0), color=(0, 0, 0), lifetime=None, parent=-1, parent_link=BASE_LINK, text_size=1):
    """a copy from pybullet.util to enable text size"""
    return p.addUserDebugText(str(text), textPosition=position, textColorRGB=color, textSize=text_size,
                              lifeTime=0, parentObjectUniqueId=parent, parentLinkIndex=parent_link,
                              physicsClientId=CLIENT)

def draw_element(node_points, element, color=(1, 0, 0)):
    n1, n2 = element.node_ids
    p1 = node_points[n1]
    p2 = node_points[n2]
    return add_line(p1, p2, color=color[:3])

def is_ground(element, ground_nodes):
    return any(n in ground_nodes for n in element.node_ids)

def draw_model(elements, node_points, ground_nodes):
    handles = []
    for element in elements:
        color = (0, 0, 1) if is_ground(element, ground_nodes) else (1, 0, 0)
        handles.append(draw_element(node_points, element, color=color))
    return handles

def load_end_effector():
    root_directory = os.path.dirname(os.path.abspath(__file__))
    with HideOutput():
        ee = load_pybullet(os.path.join(root_directory, END_EFFECTOR_PATH), fixed_base=True)
    return ee

def get_tip_from_ee_base(ee):
    world_from_tool_base = get_link_pose(ee, link_from_name(ee, EE_TOOL_BASE_LINK))
    world_from_tool_tip = get_link_pose(ee, link_from_name(ee, EE_TOOL_TIP))
    return multiply(invert(world_from_tool_tip), world_from_tool_base)

def get_ee_collision_fn(body, obstacles, attachments=[], **kwargs):
    # TODO: convert most of these to keyword arguments
    moving_bodies = [body] + [attachment.child for attachment in attachments]
    if obstacles is None:
        obstacles = list(set(get_bodies()) - set(moving_bodies))
    check_body_pairs = list(product(moving_bodies, obstacles))  # + list(combinations(moving_bodies, 2))

    def collision_fn(pose):
        set_pose(body, pose)
        for attachment in attachments:
            attachment.assign()
        return any(pairwise_collision(*pair, **kwargs) for pair in check_body_pairs)
    return collision_fn

# sample fns
################################
EEDirection = namedtuple('EEDirection', ['phi', 'theta'])

def make_print_pose(phi, theta, ee_yaw=0):
    return multiply(Pose(euler=Euler(roll=theta, pitch=0, yaw=phi)),
                    Pose(euler=Euler(yaw=ee_yaw)))

def cmap_id2angle(id, return_direction_vector=False):
    j = id % THETA_DISC
    i = (id - j) / THETA_DISC
    phi = -np.pi + i*(2*np.pi/PHI_DISC)
    theta = -np.pi + j*(2*np.pi/THETA_DISC)
    if return_direction_vector:
        return make_print_pose(phi, theta, 0)
    else:
        return phi, theta

def sample_ee_yaw():
    return random.uniform(-np.pi, np.pi)

def get_collision_fn(body, joints, obstacles, attachments, self_collisions, disabled_collisions,
                     custom_limits={}, **kwargs):
    """copy from pybullet.utils to allow dynmic collision objects"""
    # TODO: convert most of these to keyword arguments
    check_link_pairs = get_self_link_pairs(body, joints, disabled_collisions) if self_collisions else []
    moving_bodies = [body] + [attachment.child for attachment in attachments]
    if obstacles is None:
        obstacles = list(set(get_bodies()) - set(moving_bodies))
    check_body_pairs = list(product(moving_bodies, obstacles))  # + list(combinations(moving_bodies, 2))
    lower_limits, upper_limits = get_custom_limits(body, joints, custom_limits)

    # TODO: maybe prune the link adjacent to the robot
    # TODO: test self collision with the holding
    def collision_fn(q, dynamic_obstacles=[]):
        if not all_between(lower_limits, q, upper_limits):
            return True
        set_joint_positions(body, joints, q)
        for attachment in attachments:
            attachment.assign()
        if dynamic_obstacles:
            check_body_pairs.extend(list(product(moving_bodies, dynamic_obstacles)))
        for link1, link2 in check_link_pairs:
            # Self-collisions should not have the max_distance parameter
            if pairwise_link_collision(body, link1, body, link2): #, **kwargs):
                return True
        return any(pairwise_collision(*pair, **kwargs) for pair in check_body_pairs)
    return collision_fn

def check_ee_element_collision(ee_body, way_points, phi, theta, exist_e_body=None, static_bodies=[]):
    ee_yaw = sample_ee_yaw()
    for way_pt in way_points:
        ee_tip_from_base = get_tip_from_ee_base(ee_body)
        world_from_ee_tip = multiply(Pose(point=Point(*way_pt)), make_print_pose(phi, theta, ee_yaw))
        world_from_ee_base = multiply(world_from_ee_tip, ee_tip_from_base)
        set_pose(ee_body, world_from_ee_base)
        if exist_e_body:
            if pairwise_collision(ee_body, exist_e_body) != 0:
                return True
        for static_body in static_bodies:
            if pairwise_collision(ee_body, static_body) != 0:
                return True
    return False

def check_valid_kinematics(robot, way_points, phi, theta, collision_fn):
    ee_yaw = sample_ee_yaw()
    for way_pt in way_points:
        world_from_ee_tip = multiply(Pose(point=Point(*way_pt)), make_print_pose(phi, theta, ee_yaw))
        conf = sample_tool_ik(robot, world_from_ee_tip)
        if not conf or collision_fn(conf):
            # conf not exist or collision
            return False
    return True

def check_exist_valid_kinematics(assembly_network, e_id, robot, cmap, collision_fn):
    assert(len(cmap) == PHI_DISC * THETA_DISC)
    p1, p2 = assembly_network.get_end_points(e_id)
    way_points = interpolate_straight_line_pts(p1, p2, WAYPOINT_DISC_LEN)
    free_cmap_ids = [i for i in range(len(cmap)) if cmap[i] == 1]

    st_time = time.time()
    while True:
        direction_id = random.choice(free_cmap_ids)
        phi, theta = cmap_id2angle(direction_id)
        if check_valid_kinematics(robot, way_points, phi, theta, collision_fn):
            return True
        if time.time() - st_time > KINEMATICS_CHECK_TIMEOUT:
            return False

def interpolate_straight_line_pts(p1, p2, disc_len):
    p1 = np.array(p1)
    p2 = np.array(p2)
    e_len = np.linalg.norm(p1 - p2)
    advance = np.append(np.arange(0, e_len, disc_len), e_len)
    return map(tuple, [p1 + t*(p2-p1)/e_len for t in advance])

def generate_way_point_poses(p1, p2, phi, theta, ee_yaw, disc_len):
    way_points = interpolate_straight_line_pts(p1, p2, disc_len)
    return [multiply(Pose(point=pt), make_print_pose(phi, theta, ee_yaw)) for pt in way_points]

def update_collision_map(assembly_network, ee_body, print_along_e_id, exist_e_id, print_along_cmap, static_bodies=[], check_ik=False, robot=None, collision_fn=None):
    """
    :param print_along_e_id: element id that end effector is printing along
    :param exist_e_id: element that is assumed printed, checked against
    :param print_along_cmap: print_along_element's collsion map, a list of bool, 
        entry = 1 means collision-free (still available),  entry=0 means not feasible 
    :return: 
    """
    assert(len(print_along_cmap) == PHI_DISC * THETA_DISC)
    p1, p2 = assembly_network.get_end_points(print_along_e_id)
    way_points = interpolate_straight_line_pts(p1, p2, WAYPOINT_DISC_LEN)
    if exist_e_id != print_along_e_id:
        exist_e_body = assembly_network.get_element_body(exist_e_id)
    else:
        exist_e_body = None
    # assert(has_body(exist_e_body))

    for i, c_val in enumerate(print_along_cmap):
        if c_val == 1:
            phi, theta = cmap_id2angle(i)
            if check_ee_element_collision(ee_body, way_points, phi, theta, exist_e_body, static_bodies):
                print_along_cmap[i] = 0
            else:
                # exist feasible EE body pose, check ik
                if check_ik:
                    assert(check_ik and collision_fn and robot)
                    if not check_valid_kinematics(robot, way_points, phi, theta, collision_fn):
                        print_along_cmap[i] = 0

            # TODO: check against shrinked geoemtry only if the exist_e is in neighborhood of print_along_e

    return print_along_cmap

def draw_assembly_sequence(assembly_network, element_id_sequence, seq_poses=None,
                           line_width=10, text_size=1, time_step=0.5, direction_len=0.005):
    handles = []
    for k in element_id_sequence.keys():
        e_id = element_id_sequence[k]
        # n1, n2 = assembly_network.assembly_elements[e_id].node_ids
        # e_body = assembly_network.assembly_elements[e_id].element_body
        # p1 = assembly_network.assembly_joints[n1].node_point
        # p2 = assembly_network.assembly_joints[n2].node_point
        p1, p2 = assembly_network.get_end_points(e_id)
        e_mid = (np.array(p1) + np.array(p2)) / 2

        seq_ratio = float(k)/(len(element_id_sequence)-1)
        color = np.array([0, 0, 1])*(1-seq_ratio) + np.array([1,0,0])*seq_ratio
        handles.append(add_line(p1, p2, color=tuple(color), width=line_width))
        handles.append(add_text(str(k), position=e_mid, text_size=text_size))

        if seq_poses is not None:
            assert(seq_poses.has_key(k))
            for ee_dir in seq_poses[k]:
                assert(isinstance(ee_dir, EEDirection))
                cmap_pose = multiply(Pose(point=e_mid), make_print_pose(ee_dir.phi, ee_dir.theta))
                origin_world = tform_point(cmap_pose, np.zeros(3))
                axis = np.zeros(3)
                axis[2] = 1
                axis_world = tform_point(cmap_pose, direction_len*axis)
                handles.append(add_line(origin_world, axis_world, color=axis))
                # handles.append(draw_pose(cmap_pose, direction_len))

        time.sleep(time_step)

def write_seq_json(assembly_network, element_seq, seq_poses, file_name):
    root_directory = os.path.dirname(os.path.abspath(__file__))
    file_dir = os.path.join(root_directory, 'results')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = os.path.join(file_dir, file_name + '.json')
    if not os.path.exists(file_path):
        open(file_path, "w+").close()

    data = OrderedDict()
    data['assembly_type'] = 'extrusion'
    data['file_name'] = file_name
    data['element_number'] = assembly_network.get_size_of_elements()
    data['support_number'] = assembly_network.get_size_of_grounded_elements()

    # TODO: base_center_pt, maybe we don't need it at all?

    data['sequenced_elements'] = []
    for i in element_seq.keys():
        e_id = element_seq[i]
        e_data = OrderedDict()
        e_data['order_id'] = i
        e_data['element_id'] = e_id # id in the network

        e_data['start_pt'], e_data['end_pt'] = map(list, assembly_network.get_end_points(e_id))
        # e_data['element_type'] = 'support', 'connect', 'create'

        # feasible ee directions
        # TODO: convert cmaps to ee pose outside, and pass in pose, convert it into Euler here
        e_data['feasible_directions'] = []
        for ee_direction in seq_poses[i]:
            assert(isinstance(ee_direction, EEDirection))
            e_data['feasible_directions'].append({'phi': ee_direction.phi, 'theta': ee_direction.theta})

        data['sequenced_elements'].append(e_data)

    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)

def read_seq_json(file_name):
    """parse existing sequence plan, return a dict of EE pose lists, indexed by sequence order, NOT element id"""
    root_directory = os.path.dirname(os.path.abspath(__file__))
    file_dir = os.path.join(root_directory, 'results')
    file_path = os.path.join(file_dir, file_name + '.json')
    if not os.path.exists(file_dir) or not os.path.exists(file_path):
        return None, None
    try:
        with open(file_path, 'r') as f:
            json_data = json.loads(f.read())

        seq_poses = {}
        element_seq = {}
        assert(json_data.has_key('sequenced_elements'))
        for e in json_data['sequenced_elements']:
            element_seq[e['order_id']] = e['element_id']
            seq_poses[e['order_id']] = \
                [EEDirection(phi=pose['phi'], theta=pose['theta']) for pose in e['feasible_directions']]

        print('sequence plan parse: {}'.format(file_path))
        return element_seq, seq_poses
    except Exception as e:
        print('No existing sequence plan found, return False: {}'.format(e))
        return None, None