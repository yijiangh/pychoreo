from __future__ import print_function

import json
import os
import random
import time
from copy import copy
from collections import defaultdict
import numpy as np

from pybullet_planning import set_pose, multiply, pairwise_collision, get_collision_fn, joints_from_names, \
    get_disabled_collisions, interpolate_poses, get_moving_links, get_body_body_disabled_collisions
from pybullet_planning import RED

from pychoreo.utils.stream_utils import get_enumeration_pose_generator
from pychoreo.cartesian_planner.cartesian_process import prune_ee_feasible_directions
from pychoreo_examples.extrusion.stream import extrusion_ee_pose_gen_fn

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

def add_collision_fns_from_seq(robot, ik_joint_names, cart_process_dict, element_seq, element_bodies,
        domain_size, ee_pose_map_fn, ee_body, reverse_flags=None, sample_time=5,
        tool_from_root=None, workspace_bodies=[], ws_disabled_body_link_names={}, static_obstacles=[],
        self_collisions=True, disabled_self_collision_link_names=[], pos_step_size=0.003):

    assert len(cart_process_dict) == len(element_seq)
    assert len(element_bodies) == len(element_seq)
    if reverse_flags:
        assert len(reverse_flags) == len(element_seq)

    # converting link names to robot pb links
    ik_joints = joints_from_names(robot, ik_joint_names)
    disabled_collisions = get_disabled_collisions(robot, disabled_self_collision_link_names)
    ws_disabled_collisions = set()
    for ws_body in workspace_bodies:
        ws_disabled_collisions.update(get_body_body_disabled_collisions(robot, ws_body, ws_disabled_body_link_names))
    # print(disabled_collisions)
    # print(ws_disabled_collisions)

    # collision objects might be modeled in the URDF as robot links
    # TODO: get collision object links in the URDF from compas_fab

    built_obstacles = copy(static_obstacles)
    e_fmaps = {e : [1 for _ in range(domain_size)] for e in element_seq}
    for element in element_seq:
        print('checking E#{}'.format(element))
        st_time = time.time()
        while time.time() - st_time < sample_time:
            e_fmaps[element] = prune_ee_feasible_directions(cart_process_dict[element],
                                    e_fmaps[element], ee_pose_map_fn, ee_body, tool_from_root=tool_from_root,
                                    collision_objects=built_obstacles, workspace_bodies=workspace_bodies, check_ik=False)
            if sum(e_fmaps[element]) > 0:
                break
        assert sum(e_fmaps[element]) > 0, 'E#{} feasible map empty, precomputed sequence should have a feasible ee pose range!'.format(element)
        print('E#{} valid, feasible poses: {}'.format(element, sum(e_fmaps[element])))

        # use pruned direction set to gen ee path poses
        enum_gen_fn = get_enumeration_pose_generator([ee_pose_map_fn(i) for i, is_feasible in enumerate(e_fmaps[element]) if is_feasible])
        ee_pose_gen_fn = extrusion_ee_pose_gen_fn(cart_process_dict[element].path_points,
            enum_gen_fn, interpolate_poses, pos_step_size=pos_step_size)
        cart_process_dict[element].ee_pose_gen_fn = ee_pose_gen_fn

        # use sequenced elements for collision objects
        built_obstacles = built_obstacles + [element_bodies[element]]
        cart_process_dict[element].collision_fn = get_collision_fn(robot, ik_joints, built_obstacles,
                                         attachments=[], check_self_collisions=self_collisions,
                                         disabled_self_collision_link_pairs=disabled_collisions,
                                         ws_bodies=workspace_bodies,
                                         ws_disabled_body_link_pairs=ws_disabled_collisions,
                                         custom_limits={})
    return [cart_process_dict[e] for e in element_seq], e_fmaps
