import time
import random
from itertools import product
from copy import copy
import numpy as np

from pybullet_planning import INF, RED
from pybullet_planning import set_pose, multiply, pairwise_collision, get_collision_fn, joints_from_names, \
    get_disabled_collisions, interpolate_poses, get_moving_links, get_body_body_disabled_collisions, interval_generator
from pybullet_planning import Point, Pose, Euler, unit_pose, quat_angle_between, angle_between, matrix_from_quat
from pybullet_planning import get_collision_fn, get_floating_body_collision_fn

from pychoreo.utils.stream_utils import get_enumeration_pose_generator
from pychoreo.utils.general_utils import is_any_empty
from pychoreo.process_model.cartesian_process import CartesianProcess, CartesianSubProcess
from pychoreo.process_model.gen_fn import CartesianPoseGenFn

from pychoreo_examples.extrusion.utils import is_ground
from pychoreo_examples.extrusion.trajectory import PrintTrajectory, PrintBufferTrajectory


################################################

def get_extrusion_ee_pose_compose_fn(ee_pose_interp_fn, approach_distance=0.05, **kwargs):
    def pose_compose_fn(ee_orient, base_path_pts):
        extrude_from_approach = Pose(Point(z=-approach_distance))
        extrude_landmarks = [multiply(Pose(point=Point(*pt)), ee_orient) for pt in base_path_pts]
        ee_landmarks = []
        # approach
        ee_landmarks.append([multiply(extrude_landmarks[0], extrude_from_approach), extrude_landmarks[0]])
        # extrude
        ee_landmarks.append(extrude_landmarks)
        # retreat
        ee_landmarks.append([extrude_landmarks[-1], multiply(extrude_landmarks[-1], extrude_from_approach)])

        process_path = []
        for end_pts_path in ee_landmarks:
            sub_path = []
            for i in range(len(end_pts_path)-1):
                sub_path.extend(ee_pose_interp_fn(end_pts_path[i], end_pts_path[i+1], **kwargs))
            process_path.append(sub_path)
        return process_path
    return pose_compose_fn

def get_ee_pose_enumerate_map_fn(roll_disc, pitch_disc):
    def ee_pose_map_fn(id, yaw=None):
        j = id % roll_disc
        i = (id - j) / pitch_disc
        roll = -np.pi + i*(2*np.pi/roll_disc)
        pitch = -np.pi + j*(2*np.pi/pitch_disc)
        yaw = random.uniform(-np.pi, np.pi) if yaw is None else yaw
        return multiply(Pose(euler=Euler(roll=roll, pitch=pitch)), Pose(euler=Euler(yaw=yaw)))
    return ee_pose_map_fn

def find_closest_map_id_to_pose_dir(target_pose, domain_size, ee_pose_map_fn):
    target_dir = get_ee_pointing_direction(target_pose)
    def distance_to_dir(pose_id):
        pose = ee_pose_map_fn(pose_id, yaw=0)
        ee_dir = get_ee_pointing_direction(pose)
        return angle_between(ee_dir, target_dir)
    return min(list(range(domain_size)), key=distance_to_dir)

################################################
# preference cost evaluation fn

def get_ee_pointing_direction(ee_pose):
    quat = ee_pose[1]
    z_axis = matrix_from_quat(quat)[:, 2]
    return z_axis

def get_create_preference_eval_fn(element_dir, lower_cost, upper_cost):
    # TODO: prefer having the cooling pipe parellel to the element direction
    # prefer the directions that are closer to the element direction
    def cost_val_fn(ee_poses):
        # print('create cost: ', abs(angle_between(element_dir, -1 * get_ee_pointing_direction(ee_poses[1][0]))) / np.pi)
        pf_cost = lower_cost + (upper_cost - lower_cost) * \
            angle_between(element_dir, -1 * get_ee_pointing_direction(ee_poses[1][0])) / np.pi
        # print('create cost: ', pf_cost)
        return pf_cost
    return cost_val_fn

def get_connect_preference_eval_fn(element_dir, lower_cost, upper_cost):
    # prefer directions that are perpendicular to the element direction
    # TODO: prefer having the cooling pipe parellel to the element direction
    def cost_val_fn(ee_poses):
        # print('connect cost: ', abs(angle_between(element_dir, get_ee_pointing_direction(ee_poses[1][0])) - np.pi/2) / np.pi)
        pf_cost = lower_cost + (upper_cost - lower_cost) * \
            abs(angle_between(element_dir, get_ee_pointing_direction(ee_poses[1][0])) - np.pi/2) / np.pi
        # print('connect cost: ', pf_cost)
        return pf_cost
    return cost_val_fn

################################################
# cartesian process building

def build_extrusion_cartesian_process_sequence(
        element_seq, element_bodies, node_points, ground_nodes,
        robot, ik_joint_names, sample_ik_fn, ee_body,
        ee_fmaps=None, ee_pose_map_fn=None,
        roll_disc=10, pitch_disc=10, yaw_sample_size=10,
        sample_time=5, approach_distance=0.01, linear_step_size=0.003, tool_from_root=None,
        self_collisions=True, disabled_collisions={},
        obstacles=None, extra_disabled_collisions={},
        reverse_flags=None, verbose=False, max_attempts=2):

    # make sure we don't modify the obstacle list by accident
    built_obstacles = copy(obstacles) if obstacles else []

    ik_joints = joints_from_names(robot, ik_joint_names)
    if not reverse_flags: reverse_flags = [False for _ in len(element_seq)]

    if isinstance(ee_fmaps,dict) and callable(ee_pose_map_fn):
        use_parsed_fmaps = True
        print('Using parsed ee_fmaps to build extrusion Cartesian processes.')
    else:
        use_parsed_fmaps = False
        domain_size = roll_disc * pitch_disc
        ee_fmaps = {e : [1 for _ in range(domain_size)] for e in element_seq}
        ee_pose_map_fn = get_ee_pose_enumerate_map_fn(roll_disc, pitch_disc)

    # TODO: remove later
    # json_data = {}

    node_visited_valence = {}
    cart_proc_seq = []
    for seq_id, element in enumerate(element_seq):
        n1, n2 = element
        base_path_pts = [node_points[n1], node_points[n2]]
        if reverse_flags[element]:
            base_path_pts = base_path_pts[::-1]

        # subprocess tagging
        n1_visited = n1 in node_visited_valence
        n2_visited = n2 in node_visited_valence
        if n1_visited:
            node_visited_valence[n1] += 1
        else:
            node_visited_valence[n1] = 0
        if n2_visited:
            node_visited_valence[n2] += 1
        else:
            node_visited_valence[n2] = 0
        if is_ground(element, ground_nodes):
            extrusion_tag = 'ground'
        else:
            assert n1_visited or n2_visited, 'this element is floating!'
            extrusion_tag = 'connect' if n1_visited and n2_visited else 'create'

        # TODO: move these parameters to arguement
        # a "distance function" measuring the distance between the ideal direction and the sampled one
        lower_cost = 1.0
        upper_cost = 1000.0
        element_dir = base_path_pts[1] - base_path_pts[0]
        if extrusion_tag == 'create':
            eval_preference_cost_fn = get_create_preference_eval_fn(element_dir, lower_cost, upper_cost)
        elif extrusion_tag == 'connect':
            eval_preference_cost_fn = get_connect_preference_eval_fn(element_dir, lower_cost, upper_cost)
        else:
            eval_preference_cost_fn = lambda poses : 1.0

        extrusion_compose_fn = get_extrusion_ee_pose_compose_fn(interpolate_poses, approach_distance=approach_distance, pos_step_size=linear_step_size)
        full_path_pts = extrusion_compose_fn(unit_pose(), base_path_pts)

        # use sequenced elements for collision objects
        collision_fn = get_collision_fn(robot, ik_joints, built_obstacles,
                                        attachments=[], self_collisions=self_collisions,
                                        disabled_collisions=disabled_collisions,
                                        custom_limits={})

        if verbose : print('----\nPruning candidate poses for E#{}'.format(element))
        if not is_ground(element, ground_nodes):
            # before_sum = sum(e_fmaps[element])
            # diagnosis = True if seq_id == 19 else False
            if not use_parsed_fmaps:
                if extrusion_tag == 'connect':
                    # prune the one that causes the EE to collide with the element
                    for k in range(domain_size):
                        angle_to_element = angle_between(element_dir, get_ee_pointing_direction(ee_pose_map_fn(k)))
                        # TODO: move these parameters to arguement
                        # TODO: maybe try np.pi / 3?
                        if angle_to_element < np.pi/6 or angle_to_element > np.pi* 5/6:
                            ee_fmaps[element][k] = 0
                    # print('connect, after sum: {}/{}'.format(sum(e_fmaps[element]), before_sum))
                elif extrusion_tag == 'create':
                    for i in range(domain_size):
                        angle_to_element = angle_between(element_dir, get_ee_pointing_direction(ee_pose_map_fn(i)))
                        if angle_to_element < (1/3) * np.pi:
                            ee_fmaps[element][i] = 0
                #     print('create, after sum: {}/{}'.format(sum(e_fmaps[element]), before_sum))
                diagnosis = False
                ee_fmaps[element] = prune_ee_feasible_directions(full_path_pts,
                                                                ee_fmaps[element], ee_pose_map_fn, ee_body,
                                                                obstacles=obstacles,
                                                                tool_from_root=tool_from_root,
                                                                check_ik=True, sample_ik_fn=sample_ik_fn, collision_fn=collision_fn,
                                                                sub_process_ids=[(1,[0, int(len(full_path_pts[1])/2.0), len(full_path_pts[1])-1])], max_attempts=max_attempts,
                                                                diagnosis=diagnosis)
            # TODO: put this back later
            # assert sum(e_fmaps[element]) > 0, 'E#{} feasible map empty, precomputed sequence should have a feasible ee pose range!'.format(element)

            # use pruned direction set to gen ee path poses
            direction_poses = [ee_pose_map_fn(i) for i, is_feasible in enumerate(ee_fmaps[element]) if is_feasible]

            # json_data[extrusion_tag] = {}
            # json_data[extrusion_tag]['node_points'] = [pt.tolist() for pt in base_path_pts]
            # json_data[extrusion_tag]['element_dir'] = element_dir.tolist()
            # json_data[extrusion_tag]['direction_poses'] = [get_ee_pointing_direction(eep).tolist() for eep in direction_poses]
        else:
            # we only allow negative z direction for grounded elements
            direction_poses = [Pose(euler=Euler(pitch=np.pi))]
            domain_size = len(ee_fmaps[element])
            ee_fmaps[element] = [0 for _ in range(domain_size)]
            for dir_pose in direction_poses:
                ee_fmaps[element][find_closest_map_id_to_pose_dir(dir_pose, domain_size, ee_pose_map_fn)] = 1

        if yaw_sample_size < INF:
            yaw_gen = interval_generator([-np.pi]*yaw_sample_size, [np.pi]*yaw_sample_size)
            yaw_samples = next(yaw_gen)
            candidate_poses = [multiply(dpose, Pose(euler=Euler(yaw=yaw))) for dpose, yaw in product(direction_poses, yaw_samples)]
            if verbose : print('{}/{}: E#{} valid, candidate poses: {}, build enumeration sampler'.format(seq_id, len(element_seq)-1, element, len(candidate_poses)))
            orient_gen_fn = get_enumeration_pose_generator(candidate_poses, shuffle=True)
        else:
            def get_yaw_generator(base_poses, dir_attempts=3):
                while True:
                    dpose = random.choice(base_poses)
                    for _ in range(dir_attempts):
                        yaw = random.uniform(-np.pi, +np.pi)
                        yield multiply(dpose, Pose(euler=Euler(yaw=yaw)))
            if verbose : print('{}/{}: E#{} valid, candidate direction poses: {}, build inf sampler'.format(seq_id, len(element_seq)-1, element, len(direction_poses)))
            orient_gen_fn = get_yaw_generator(direction_poses)

        pose_gen_fn = CartesianPoseGenFn(orient_gen_fn, extrusion_compose_fn, base_path_pts=base_path_pts)

        # build three sub-processes: approach, extrusion, retreat
        # they share the same collision_fn
        extrusion_sub_procs = [CartesianSubProcess(sub_process_name='approach-extrude', collision_fn=collision_fn),
                               CartesianSubProcess(sub_process_name='extrude', collision_fn=collision_fn),
                               CartesianSubProcess(sub_process_name='extrude-retreat', collision_fn=collision_fn)]

        extrusion_sub_procs[0].trajectory = PrintBufferTrajectory(robot, ik_joints, None, element,
            is_reverse=reverse_flags[element], tag='approach')
        extrusion_sub_procs[1].trajectory = PrintTrajectory(robot, ik_joints, None, element,
            is_reverse=reverse_flags[element], tag=extrusion_tag)
        extrusion_sub_procs[2].trajectory = PrintBufferTrajectory(robot, ik_joints, None, element,
            is_reverse=reverse_flags[element], tag='retreat')

        # TODO: add pointwise collision fn to prevent last conf collides with the element currently printing

        process_name = 'extrusion-E{}'.format(element)
        cart_process = CartesianProcess(process_name=process_name,
            robot=robot, ik_joint_names=ik_joint_names,
            sub_process_list=extrusion_sub_procs,
            ee_pose_gen_fn=pose_gen_fn, sample_ik_fn=sample_ik_fn,
            element_identifier=element, preference_cost_eval_fn=eval_preference_cost_fn)

        cart_proc_seq.append(cart_process)
        built_obstacles = built_obstacles + [element_bodies[element]]

    # # TODO: remove later
    # import json
    # with open(r'C:\Users\yijiangh\Desktop\dir_data.json', 'w') as outfile:
    #     json.dump(json_data, outfile)

    return cart_proc_seq, ee_fmaps

##################################################

def prune_ee_feasible_directions(way_poses, free_pose_map, ee_pose_map_fn, ee_body, max_attempts=5,
                                 self_collisions=True, disabled_collisions={},
                                 obstacles=[], extra_disabled_collisions={},
                                 sub_process_ids=None,
                                 tool_from_root=None,
                                 check_ik=False, sample_ik_fn=lambda x : [],
                                 collision_fn=lambda x : False, diagnosis=False):

    ee_collision_fn = get_floating_body_collision_fn(ee_body, obstacles,
                                                     disabled_collisions=disabled_collisions)

    if not sub_process_ids:
        sub_process_ids = list(zip(range(len(way_poses)), [list(range(len(sp_poses))) for sp_poses in way_poses]))
    else:
        for sp_id, sp_pair in enumerate(sub_process_ids):
            if len(sp_pair[1]) == 0:
                sub_process_ids[sp_id] = (sp_pair[0], [pt_id for pt_id in range(len(way_poses[sp_pair[0]]))])
                # TODO: sanity check specified pt_ids

    fmap_ids = list(range(len(free_pose_map)))
    random.shuffle(fmap_ids)
    for i in fmap_ids:
        if free_pose_map[i]:
            for _ in range(max_attempts):
                is_colliding = False
                direction_pose = ee_pose_map_fn(i)
                oriented_way_poses = [[(pt[0], direction_pose[1]) for pt in sp_way_points] for sp_way_points in way_poses]

                for sp_id, pt_ids in sub_process_ids:
                    for pt_id in pt_ids:
                        # print('checking: {}:{}'.format(sp_id, pt_ids))
                        tcp_pose = oriented_way_poses[sp_id][pt_id]
                        # transform TCP to EE tool base link
                        if tool_from_root:
                            root_pose = multiply(tcp_pose, tool_from_root)
                        # check pairwise collision between the EE and collision objects
                        is_colliding = ee_collision_fn(root_pose)
                        if is_colliding:
                            break
                        if not is_colliding and check_ik: # and pt_id in [int(len(pt_ids)/2.0)]: # [0, int(len(pt_ids)/2.0), len(pt_ids)-1]:
                            jt_list = sample_ik_fn(tcp_pose)
                            jt_list = [jts for jts in jt_list \
                                if jts and not collision_fn(jts, diagnosis=diagnosis)]
                            is_colliding = is_any_empty(jt_list)
                        if is_colliding:
                            break
                    if is_colliding:
                        break
                if not is_colliding:
                    # solution found!
                    break
            free_pose_map[i] = 0 if is_colliding else 1
    return free_pose_map
