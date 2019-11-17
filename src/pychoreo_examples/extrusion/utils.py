from __future__ import print_function

import json
import os
import random
import time
from copy import copy
from collections import defaultdict
from itertools import product
import numpy as np

from pybullet_planning import set_pose, multiply, pairwise_collision, get_collision_fn, joints_from_names, \
    get_disabled_collisions, interpolate_poses, get_moving_links, get_body_body_disabled_collisions, interval_generator
from pybullet_planning import RED, Pose, Euler

from pychoreo.utils.stream_utils import get_enumeration_pose_generator
from pychoreo.process_model.cartesian_process import prune_ee_feasible_directions
from pychoreo.process_model.gen_fn import CartesianPoseGenFn
from pychoreo_examples.extrusion.stream import extrusion_ee_pose_gen_fn

##################################################

def is_ground(element, ground_nodes):
    return any(n in ground_nodes for n in element)


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

##################################################

def max_valence_extrusion_direction_routing(element_sequence, elements, grounded_node_ids):
    # based on valence now, can be
    reverse_flags = [False for e in element_sequence]

        # if not csp.net.is_element_grounded(check_e_id):
        #     ngbh_e_ids = rec_seq.intersection(csp.net.get_element_neighbor(check_e_id))
        #     shared_node = set()
        #     for n_e in ngbh_e_ids:
        #         shared_node.update([csp.net.get_shared_node_id(check_e_id, n_e)])
        #     shared_node = list(shared_node)
        # else:
        #     shared_node = [v_id for v_id in csp.net.get_element_end_point_ids(check_e_id)
        #                    if csp.net.assembly_joints[v_id].is_grounded]
        # assert(shared_node)

    return reverse_flags

def add_collision_fns_from_seq(robot, ik_joints, cart_process_dict,
        element_seq, element_bodies,
        domain_size, ee_pose_map_fn, ee_body,
        yaw_sample_size=10, sample_time=5, approach_distance=0.01, linear_step_size=0.003, tool_from_root=None,
        self_collisions=True, disabled_collisions={},
        obstacles=[], extra_disabled_collisions={},
        verbose=False):

    assert len(cart_process_dict) == len(element_seq)
    assert len(element_bodies) == len(element_seq)

    built_obstacles = copy(obstacles)
    e_fmaps = {e : [1 for _ in range(domain_size)] for e in element_seq}
    for element in element_seq:
        if verbose : print('checking E#{}'.format(element))
        st_time = time.time()
        while time.time() - st_time < sample_time:
            e_fmaps[element] = prune_ee_feasible_directions(cart_process_dict[element],
                                    e_fmaps[element], ee_pose_map_fn, ee_body,
                                    obstacles=obstacles,
                                    tool_from_root=tool_from_root, check_ik=False)
            if sum(e_fmaps[element]) > 0:
                break
        assert sum(e_fmaps[element]) > 0, 'E#{} feasible map empty, precomputed sequence should have a feasible ee pose range!'.format(element)

        # use pruned direction set to gen ee path poses
        direction_poses = [ee_pose_map_fn(i) for i, is_feasible in enumerate(e_fmaps[element]) if is_feasible]
        # direct enumarator or random sampling
        # yaw_samples = np.arange(-np.pi, np.pi, 2*np.pi/yaw_sample_size)
        yaw_gen = interval_generator([-np.pi]*yaw_sample_size, [np.pi]*yaw_sample_size)
        yaw_samples = next(yaw_gen)

        candidate_poses = [multiply(dpose, Pose(euler=Euler(yaw=yaw))) for dpose, yaw in product(direction_poses, yaw_samples)]
        enum_gen_fn = get_enumeration_pose_generator(candidate_poses, shuffle=True)
        if verbose : print('E#{} valid, candidate poses: {}'.format(element, len(candidate_poses)))
        new_pose_gen_fn = extrusion_ee_pose_gen_fn(cart_process_dict[element].ee_pose_gen_fn.base_path_pts,
                                                   enum_gen_fn, interpolate_poses, approach_distance=approach_distance, pos_step_size=linear_step_size)
        cart_process_dict[element].ee_pose_gen_fn.update_gen_fn(new_pose_gen_fn)
        # cart_process_dict[element].ee_pose_gen_fn = CartesianPoseGenFn(cart_process_dict[element].ee_pose_gen_fn.base_path_pts, new_pose_gen_fn)

        # TODO: reverse info

        # use sequenced elements for collision objects
        collision_fn = get_collision_fn(robot, ik_joints, built_obstacles,
                                        attachments=[], self_collisions=self_collisions,
                                        disabled_collisions=disabled_collisions,
                                        custom_limits={})
        for sub_process in cart_process_dict[element].sub_process_list:
            sub_process.collision_fn = collision_fn

        built_obstacles = built_obstacles + [element_bodies[element]]

        # import pychoreo_examples
        # from pybullet_planning import set_pose, set_joint_positions, wait_for_user, has_gui, wait_for_user
        # viz_step = True
        # cart_process = cart_process_dict[element]
        # # ee_poses = cart_process_dict[element].sample_ee_poses(reset_iter=True)
        # print('\n$$$$$$$$$$$$$$\nenum gen fn viz')
        # cnt = 0
        # for ee_poses in cart_process.exhaust_iter():
        #     if viz_step:
        #         for sp_id, sp in enumerate(ee_poses):
        #             for ee_p in sp:
        #                 ee_p = multiply(ee_p, tool_from_root)
        #                 set_pose(ee_body, ee_p)
        #                 if has_gui(): wait_for_user()

        #     ik_sols = cart_process.get_ik_sols(ee_poses, check_collision=False)
        #     if viz_step:
        #         for sp_id, sp_jt_sols in enumerate(ik_sols):
        #             for jt_sols in sp_jt_sols:
        #                 for jts in jt_sols:
        #                     set_joint_positions(robot, ik_joints, jts)
        #                     if has_gui(): wait_for_user()
        #     cnt += 1
        #     if cnt > 3 : break

    return [cart_process_dict[e] for e in element_seq], e_fmaps
