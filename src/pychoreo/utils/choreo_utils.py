import warnings
warnings.warn("choreo_utils module deprecated", DeprecationWarning)

import os
import random
import time
import datetime
import numpy as np
import pybullet as p
from itertools import product, combinations, count
import json
from collections import OrderedDict, namedtuple

from conrob_pybullet.ss_pybullet.pybullet_tools.utils import add_line, Euler, \
    set_joint_positions, pairwise_collision, Pose, multiply, Point, HideOutput, load_pybullet, link_from_name, \
    get_link_pose, invert, get_bodies, set_pose, add_text, CLIENT, BASE_LINK, get_custom_limits, all_between, pairwise_link_collision, \
    tform_point, matrix_from_quat, MAX_DISTANCE, set_color, wait_for_user, set_camera_pose, \
    add_body_name, get_name, euler_from_quat, add_fixed_constraint, get_sample_fn, get_extend_fn, \
    get_distance_fn, get_joint_positions, check_initial_end, get_moving_links, get_links, get_moving_pairs, are_links_adjacent, \
    link_pairs_collision, draw_link_name, get_link_name, get_all_links, remove_debug, get_distance, has_gui, wait_for_duration
# from .assembly_csp import AssemblyCSP

from conrob_pybullet.ss_pybullet.motion.motion_planners.rrt_connect import birrt

DEFAULT_SCALE = 1e-3 # TODO: load different scales
EPS = 1e-5

END_EFFECTOR_PATH = '../conrob_pybullet/models/kuka_kr6_r900/urdf/extrusion_end_effector.urdf'
EE_TOOL_BASE_LINK = 'eef_base_link'
EE_TOOL_TIP = 'eef_tcp_frame'

SELF_COLLISION_ANGLE = 30.0 * (np.pi / 180)

# end effector direction discretization
THETA_DISC = 20
PHI_DISC = 20

WAYPOINT_DISC_LEN = 0.01 # meter
KINEMATICS_CHECK_TIMEOUT = 2


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

def get_self_link_pairs(body, joints, disabled_collisions=set(), only_moving=True):
    moving_links = get_moving_links(body, joints)
    fixed_links = list(set(get_links(body)) - set(moving_links))
    check_link_pairs = list(product(moving_links, fixed_links))
    if only_moving:
        check_link_pairs.extend(get_moving_pairs(body, joints))
    else:
        check_link_pairs.extend(combinations(moving_links, 2))
    check_link_pairs = list(filter(lambda pair: not are_links_adjacent(body, *pair), check_link_pairs))
    check_link_pairs = list(filter(lambda pair: (pair not in disabled_collisions) and
                                                (pair[::-1] not in disabled_collisions), check_link_pairs))
    return check_link_pairs

def get_link_attached_body_pairs(body, attachments=[]):
    link_body_pairs = []
    body_links = get_links(body)

    for attach in attachments:
        if attach.child in [pair[1] for pair in link_body_pairs]:
            continue
        if attach.parent == body:
            body_check_links = list(filter(lambda b_l : attach.parent_link != b_l and \
                not are_links_adjacent(body, b_l, attach.parent_link), body_links))
            link_body_pairs.append((body_check_links, attach.child))

    return link_body_pairs


def draw_collision_diagnosis(pybullet_output, paint_all_others=False, viz_last_duration=-1):
    if not has_gui() and pybullet_output:
        return
    if paint_all_others:
        set_all_bodies_color()
        # for b in obstacles:
        #     set_color(b, (0,0,1,0.3))
    for u_cr in pybullet_output:
        handles = []
        b1 = get_body_from_pb_id(u_cr[1])
        b2 = get_body_from_pb_id(u_cr[2])
        l1 = u_cr[3]
        l2 = u_cr[4]
        l1_name = get_link_name(b1, l1)
        l2_name = get_link_name(b2, l2)
        print('pairwise LINK collision: Body #{0} Link #{1} - Body #{2} Link #{3}'.format(
            get_name(b1), l1_name, get_name(b2), l2_name))
        set_color(b1, (1, 0, 0, 0.2))
        set_color(b2, (0, 1, 0, 0.2))

        handles.append(add_body_name(b1))
        handles.append(add_body_name(b2))
        handles.append(draw_link_name(b1, l1))
        handles.append(draw_link_name(b2, l2))

        handles.append(add_line(u_cr[5], u_cr[6], color=(0,0,1), width=5))
        # camera_base_pt = u_cr[5]
        # camera_pt = np.array(camera_base_pt) + np.array([0.1, 0, 0.05])
        # set_camera_pose(tuple(camera_pt), camera_base_pt)

        if viz_last_duration < 0:
            wait_for_user()
        else:
            wait_for_duration(viz_last_duration)

        # restore lines and colors
        for h in handles: remove_debug(h)
        set_color(b1, (1, 1, 1, 1))
        set_color(b2, (1, 1, 1, 1))


def check_ee_element_collision(ee_body, way_points, phi, theta, exist_e_body=None, static_bodies=[], in_print_collision_obj=[]):
    ee_yaw = sample_ee_yaw()
    ee_tip_from_base = get_tip_from_ee_base(ee_body)
    print_orientation = make_print_pose(phi, theta, ee_yaw)

    # TODO: this assuming way points form a line
    assert(len(way_points) >= 2)
    e_dir = np.array(way_points[1]) - np.array(way_points[0])
    (_, print_quat) = print_orientation
    print_z_dir = matrix_from_quat(print_quat)[:,2]

    e_dir = e_dir / np.linalg.norm(e_dir)
    print_z_dir = print_z_dir / np.linalg.norm(print_z_dir)
    angle = np.arccos(np.clip(np.dot(e_dir, print_z_dir), -1.0, 1.0))
    if angle > np.pi - SELF_COLLISION_ANGLE:
        return True

    # TODO: make this more formal...
    if in_print_collision_obj:
        way_pt = way_points[-1]
        world_from_ee_tip = multiply(Pose(point=Point(*way_pt)), print_orientation)
        world_from_ee_base = multiply(world_from_ee_tip, ee_tip_from_base)
        set_pose(ee_body, world_from_ee_base)
        for ip_obj in in_print_collision_obj:
            if pairwise_collision(ee_body, ip_obj) != 0:
                return True

    for way_pt in way_points:
        world_from_ee_tip = multiply(Pose(point=Point(*way_pt)), print_orientation)
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
    print_orientation = make_print_pose(phi, theta, ee_yaw)
    for way_pt in way_points:
        world_from_ee_tip = multiply(Pose(point=Point(*way_pt)), print_orientation)
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

def interpolate_poses_by_num(pose1, pose2, num_steps=10):
    pos1, quat1 = pose1
    pos2, quat2 = pose2
    for i in range(num_steps):
        fraction = float(i) / num_steps
        pos = (1-fraction)*np.array(pos1) + fraction*np.array(pos2)
        quat = p.getQuaternionSlerp(quat1, quat2, interpolationFraction=fraction)
        #quat = quaternion_slerp(quat1, quat2, fraction=fraction)
        yield (pos, quat)
    yield pose2

def interpolate_straight_line_pts(p1, p2, disc_len):
    p1 = np.array(p1)
    p2 = np.array(p2)
    e_len = np.linalg.norm(p1 - p2)
    advance = np.append(np.arange(0, e_len, disc_len), e_len)
    if abs(advance[-1] - advance[-2]) < EPS:
        advance = np.delete(advance, -2)
    return [tuple(p1 + t*(p2-p1)/e_len) for t in advance]

def interpolate_cartesian_poses(pose_1, pose_2, disc_len, mount_link_from_tcp=None):
    p1 = np.array(pose_1[0])
    p2 = np.array(pose_2[0])
    e_len = np.linalg.norm(p1 - p2)

    advance = np.append(np.arange(0, e_len, disc_len), e_len)
    if abs(advance[-1] - advance[-2]) < EPS:
        advance = np.delete(advance, -2)

    e1 = euler_from_quat(pose_1[1])
    e2 = euler_from_quat(pose_2[1])
    # TODO: slerp interpolation on poses

    world_from_tcps = [Pose(point=(p1 + t*(p2-p1)/e_len), euler=e1) for t in advance]
    if mount_link_from_tcp:
        return [multiply(world_from_tcp, invert(mount_link_from_tcp)) for world_from_tcp in world_from_tcps]
    else:
        return world_from_tcps

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
    # TODO: check n1 -> n2 and self direction
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
            if check_ee_element_collision(ee_body, way_points, phi, theta, exist_e_body, static_bodies, in_print_collision_obj=[assembly_network.get_element_body(print_along_e_id)]):
                print_along_cmap[i] = 0
            else:
                # exist feasible EE body pose, check ik
                if check_ik:
                    assert(check_ik and collision_fn and robot)
                    if not check_valid_kinematics(robot, way_points, phi, theta, collision_fn):
                        print_along_cmap[i] = 0

            # TODO: check against shrinked geoemtry only if the exist_e is in neighborhood of print_along_e

    return print_along_cmap


def check_exist_valid_kinematics_picknplace(assembly_network, e_id, robot, cmap, collision_fn):
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


def update_collision_map_picknplace(assembly_network, ee_body, print_along_e_id, exist_e_id, print_along_cmap, static_bodies=[], check_ik=False, robot=None, collision_fn=None):
    """
    :param print_along_e_id: element id that end effector is printing along
    :param exist_e_id: element that is assumed printed, checked against
    :param print_along_cmap: print_along_element's collsion map, a list of bool,
        entry = 1 means collision-free (still available),  entry=0 means not feasible
    :return:
    """
    # TODO: check n1 -> n2 and self direction
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
            if check_ee_element_collision(ee_body, way_points, phi, theta, exist_e_body, static_bodies, in_print_collision_obj=[assembly_network.get_element_body(print_along_e_id)]):
                print_along_cmap[i] = 0
            else:
                # exist feasible EE body pose, check ik
                if check_ik:
                    assert(check_ik and collision_fn and robot)
                    if not check_valid_kinematics(robot, way_points, phi, theta, collision_fn):
                        print_along_cmap[i] = 0

            # TODO: check against shrinked geoemtry only if the exist_e is in neighborhood of print_along_e

    return print_along_cmap



def check_and_draw_ee_collision(robot, static_obstacles, assembly_network, exist_element_id, check_e_id,
                                line_width=10, text_size=1, direction_len=0.005):
    ee_body = load_end_effector()
    val_cmap = np.ones(PHI_DISC * THETA_DISC, dtype=int)
    built_obstacles = static_obstacles
    built_obstacles = built_obstacles + [assembly_network.get_element_body(exist_e_id) for exist_e_id in exist_element_id]

    print('before pruning, cmaps sum: {}'.format(sum(val_cmap)))
    print('checking print #{} collision against: '.format(check_e_id))
    print(sorted(exist_element_id))
    print('obstables: {}'.format(built_obstacles))
    val_cmap = update_collision_map_batch(assembly_network, ee_body,
                                          print_along_e_id=check_e_id, print_along_cmap=val_cmap, bodies=built_obstacles)
    print('remaining feasible directions: {}'.format(sum(val_cmap)))

    # collision_fn = get_collision_fn(self.robot, get_movable_joints(self.robot), built_obstacles,
    #                                 attachments=[], self_collisions=SELF_COLLISIONS,
    #                                 disabled_collisions=self.disabled_collisions,
    #                                 custom_limits={})
    # return check_exist_valid_kinematics(self.net, val, self.robot, val_cmap, collision_fn)

    # drawing
    handles = []
    for e_id in exist_element_id:
        p1, p2 = assembly_network.get_end_points(e_id)
        handles.append(add_line(p1, p2, color=np.array([0, 0, 1]), width=line_width))

    p1, p2 = assembly_network.get_end_points(check_e_id)
    e_mid = (np.array(p1) + np.array(p2)) / 2
    handles.append(add_line(p1, p2, color=np.array([0, 0, 1]), width=line_width))

    for i in range(len(val_cmap)):
        if val_cmap[i] == 1:
            phi, theta = cmap_id2angle(i)
            cmap_pose = multiply(Pose(point=e_mid), make_print_pose(phi, theta))
            origin_world = tform_point(cmap_pose, np.zeros(3))
            axis = np.zeros(3)
            axis[2] = 1
            axis_world = tform_point(cmap_pose, direction_len*axis)
            handles.append(add_line(origin_world, axis_world, color=axis))
            # handles.append(draw_pose(cmap_pose, direction_len))


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
    data['write_time'] = str(datetime.datetime.now())
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
    assert(os.path.exists(file_dir) and os.path.exists(file_path))
    try:
        with open(file_path, 'r') as f:
            json_data = json.loads(f.read())

        seq_poses = {}
        element_seq = {}
        assert('sequenced_elements' in json_data)
        for e in json_data['sequenced_elements']:
            element_seq[e['order_id']] = e['element_id']
            seq_poses[e['order_id']] = \
                [EEDirection(phi=pose['phi'], theta=pose['theta']) for pose in e['feasible_directions']]

        print('sequence plan parse: {}'.format(file_path))
        return element_seq, seq_poses
    except Exception as e:
        print('No existing sequence plan found, return False: {}'.format(e))
        return None, None


def read_csp_log_json(file_name, specs='', log_path=None):
    if not log_path:
        root_directory = os.path.dirname(os.path.abspath(__file__))
        file_dir = os.path.join(root_directory, 'csp_log')
    else:
        file_dir = log_path

    file_path = os.path.join(file_dir, file_name + '_' + specs + '_csp_log.json')

    assert(os.path.exists(file_dir) and os.path.exists(file_path))
    with open(file_path, 'r') as f:
        json_data = json.loads(f.read())
        assert(json_data.has_key('assign_history'))
        assign_history = {}
        for k in json_data['assign_history'].keys():
            assign_history[int(k)] = json_data['assign_history'][k].values()
        # print(assign_history)
        print('csp_log parse: {}'.format(file_path))
        return assign_history



def get_body_from_pb_id(i):
    return p.getBodyUniqueId(i, physicsClientId=CLIENT)


def extract_file_name(str_key):
    key_sep = str_key.split('.')
    return key_sep[0]


def snap_sols(sols, q_guess, joint_limits, weights=None, best_sol_only=False):
    """get the best solution based on closeness to the q_guess and weighted joint diff

    Parameters
    ----------
    sols : [type]
        [description]
    q_guess : [type]
        [description]
    joint_limits : [type]
        [description]
    weights : [type], optional
        [description], by default None
    best_sol_only : bool, optional
        [description], by default False

    Returns
    -------
    lists of joint conf (list)
    or
    joint conf (list)
    """
    import numpy as np
    valid_sols = []
    dof = len(q_guess)
    if not weights:
        weights = [1.0] * dof
    else:
        assert dof == len(weights)

    for sol in sols:
        test_sol = np.ones(dof)*9999.
        for i in range(dof):
            for add_ang in [-2.*np.pi, 0, 2.*np.pi]:
                test_ang = sol[i] + add_ang
                if (test_ang <= joint_limits[i][1] and test_ang >= joint_limits[i][0] and \
                    abs(test_ang - q_guess[i]) < abs(test_sol[i] - q_guess[i])):
                    test_sol[i] = test_ang
        if np.all(test_sol != 9999.):
            valid_sols.append(test_sol.tolist())

    if len(valid_sols) == 0:
        return []

    if best_sol_only:
        best_sol_ind = np.argmin(np.sum((weights*(valid_sols - np.array(q_guess)))**2,1))
        return valid_sols[best_sol_ind]
    else:
        return valid_sols


def plan_joint_motion(body, joints, end_conf, obstacles=None, attachments=[],
                      self_collisions=True, disabled_collisions=set(), ignored_pairs=[],
                      weights=None, resolutions=None, max_distance=MAX_DISTANCE, custom_limits={}, **kwargs):

    assert len(joints) == len(end_conf)
    sample_fn = get_sample_fn(body, joints, custom_limits=custom_limits)
    distance_fn = get_distance_fn(body, joints, weights=weights)
    extend_fn = get_extend_fn(body, joints, resolutions=resolutions)
    collision_fn = get_collision_fn(body, joints, obstacles, attachments, self_collisions, disabled_collisions,
                                    custom_limits=custom_limits, max_distance=max_distance, ignored_pairs=ignored_pairs)

    start_conf = get_joint_positions(body, joints)

    if not check_initial_end(start_conf, end_conf, collision_fn):
        return None
    return birrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, **kwargs)
    #return plan_lazy_prm(start_conf, end_conf, sample_fn, extend_fn, collision_fn)
