import time
import random
from itertools import product
from copy import copy
import numpy as np

from pybullet_planning import multiply, interval_generator
from pybullet_planning import Pose, Point, Euler, INF

from pybullet_planning import INF
from pybullet_planning import set_pose, multiply, pairwise_collision, get_collision_fn, joints_from_names, \
    get_disabled_collisions, interpolate_poses, get_moving_links, get_body_body_disabled_collisions, interval_generator
from pybullet_planning import RED, Pose, Euler, unit_pose
from pybullet_planning import get_collision_fn, get_floating_body_collision_fn

from pychoreo.utils.stream_utils import get_enumeration_pose_generator
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
        return Pose(euler=Euler(roll=roll, pitch=pitch, yaw=yaw))
    return ee_pose_map_fn

################################################
# cartesian process building

def build_extrusion_cartesian_process_sequence(
        element_seq, element_bodies, node_points, ground_nodes,
        robot, ik_joint_names, sample_ik_fn, ee_body,
        roll_disc=10, pitch_disc=10, yaw_sample_size=10,
        sample_time=5, approach_distance=0.01, linear_step_size=0.003, tool_from_root=None,
        self_collisions=True, disabled_collisions={},
        obstacles=None, extra_disabled_collisions={},
        reverse_flags=None, verbose=False):

    # make sure we don't modify the obstacle list by accident
    built_obstacles = copy(obstacles) if obstacles else []

    ik_joints = joints_from_names(robot, ik_joint_names)
    if not reverse_flags: reverse_flags = [False for _ in len(element_seq)]
    domain_size = roll_disc * pitch_disc
    e_fmaps = {e : [1 for _ in range(domain_size)] for e in element_seq}
    ee_pose_map_fn = get_ee_pose_enumerate_map_fn(roll_disc, pitch_disc)

    node_visited_valence = {}
    cart_proc_seq = []
    for element in element_seq:
        n1, n2 = element
        base_path_pts = [node_points[n1], node_points[n2]]
        if reverse_flags[element]:
            base_path_pts = base_path_pts[::-1]

        extrusion_compose_fn = get_extrusion_ee_pose_compose_fn(interpolate_poses, approach_distance=approach_distance, pos_step_size=linear_step_size)
        full_path_pts = extrusion_compose_fn(unit_pose(), base_path_pts)

        if verbose : print('Pruning candidate poses for E#{}'.format(element))
        st_time = time.time()
        while time.time() - st_time < sample_time:
            e_fmaps[element] = prune_ee_feasible_directions(full_path_pts,
                                                            e_fmaps[element], ee_pose_map_fn, ee_body,
                                                            obstacles=obstacles,
                                                            tool_from_root=tool_from_root, check_ik=False,
                                                            sub_process_ids=[(1,[])])
            if sum(e_fmaps[element]) > 0:
                break
        assert sum(e_fmaps[element]) > 0, 'E#{} feasible map empty, precomputed sequence should have a feasible ee pose range!'.format(element)

        # use pruned direction set to gen ee path poses
        direction_poses = [ee_pose_map_fn(i) for i, is_feasible in enumerate(e_fmaps[element]) if is_feasible]

        if yaw_sample_size < INF:
            yaw_gen = interval_generator([-np.pi]*yaw_sample_size, [np.pi]*yaw_sample_size)
            yaw_samples = next(yaw_gen)
            candidate_poses = [multiply(dpose, Pose(euler=Euler(yaw=yaw))) for dpose, yaw in product(direction_poses, yaw_samples)]
            if verbose : print('E#{} valid, candidate poses: {}, build enumeration sampler'.format(element, len(candidate_poses)))
            orient_gen_fn = get_enumeration_pose_generator(candidate_poses, shuffle=True)
        else:
            def get_yaw_generator(base_poses):
                while True:
                    yaw = random.uniform(-np.pi, +np.pi)
                    dpose = random.choice(base_poses)
                    yield multiply(dpose, Pose(euler=Euler(yaw=yaw)))
            if verbose : print('E#{} valid, candidate direction poses: {}, build inf sampler'.format(element, len(direction_poses)))
            orient_gen_fn = get_yaw_generator(direction_poses)

        pose_gen_fn = CartesianPoseGenFn(orient_gen_fn, extrusion_compose_fn, base_path_pts=base_path_pts)

        # use sequenced elements for collision objects
        collision_fn = get_collision_fn(robot, ik_joints, built_obstacles,
                                        attachments=[], self_collisions=self_collisions,
                                        disabled_collisions=disabled_collisions,
                                        custom_limits={})

        # build three sub-processes: approach, extrusion, retreat
        # they share the same collision_fn
        extrusion_sub_procs = [CartesianSubProcess(sub_process_name='approach-extrude', collision_fn=collision_fn),
                               CartesianSubProcess(sub_process_name='extrude', collision_fn=collision_fn),
                               CartesianSubProcess(sub_process_name='extrude-retreat', collision_fn=collision_fn)]

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
            element_identifier=element)

        cart_proc_seq.append(cart_process)
        built_obstacles = built_obstacles + [element_bodies[element]]
    return cart_proc_seq, e_fmaps

##################################################

def prune_ee_feasible_directions(way_poses, free_pose_map, ee_pose_map_fn, ee_body,
                                 self_collisions=True, disabled_collisions={},
                                 obstacles=[], extra_disabled_collisions={},
                                 sub_process_ids=None,
                                 tool_from_root=None, check_ik=False):
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
            direction_pose = ee_pose_map_fn(i)
            oriented_way_poses = [[(pt[0], direction_pose[1]) for pt in sp_way_points] for sp_way_points in way_poses]
            for sp_id, pt_ids in sub_process_ids:
                for pt_id in pt_ids:
                    pose = oriented_way_poses[sp_id][pt_id]
                    # transform TCP to EE tool base link
                    if tool_from_root:
                        pose = multiply(pose, tool_from_root)
                    # check pairwise collision between the EE and collision objects
                    is_colliding = ee_collision_fn(pose)
                    if not is_colliding and check_ik:
                        raise NotImplementedError
                    if is_colliding:
                        free_pose_map[i] = 0
                        break
                    # wait_for_user()
    return free_pose_map
