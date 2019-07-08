from __future__ import print_function
import sys, os
sys.path.append('choreo/')

import cProfile
import pstats
import numpy as np
import argparse
import time, sys
import json
from collections import namedtuple

from conrob_pybullet.ss_pybullet.pybullet_tools.utils import connect, disconnect, wait_for_interrupt, LockRenderer, \
    has_gui, remove_body, set_camera_pose, get_movable_joints, set_joint_positions, \
    wait_for_duration, point_from_pose, get_link_pose, link_from_name, add_line, user_input,\
    HideOutput, load_pybullet, create_obj, draw_pose, add_body_name, get_pose, \
    pose_from_tform, invert, multiply, set_pose
from choreo.choreo_utils import draw_model, draw_assembly_sequence, write_seq_json, \
read_seq_json, cmap_id2angle, EEDirection, check_and_draw_ee_collision, \
set_cmaps_using_seq, parse_transform
from choreo.sc_cartesian_planner import divide_list_chunks, SparseLadderGraph, direct_ladder_graph_solve_picknplace
from choreo.assembly_datastructure import Brick
try:
    from conrob_pybullet.utils.ikfast.kuka_kr6_r900.ik import sample_tool_ik
except ImportError as e:
    print('\x1b[6;30;43m' + '{}, Using pybullet ik fn instead'.format(e) + '\x1b[0m')
    USE_IKFAST = False
    user_input("Press Enter to continue...")
else:
    USE_IKFAST = True

PICKNPLACE_DIRECTORY = os.path.join('..', '..', 'assembly_instances', 'picknplace')
PICKNPLACE_FILENAMES = {
    'toggle_rebar_cage_1': 'toggle_rebar_cage_1.json'
}
IRB6600_TRACK_URDF = os.path.join('..','..','conrob_pybullet','models','abb_irb6600_track','urdf','irb6600_track_toggle.urdf')
TOOL_NAME = 'eef_tcp_frame' # robot_tool0 | eef_base_link | eef_tcp_frame
GRASP_NAMES = ['pick_grasp_approach_plane', 'pick_grasp_plane', 'pick_grasp_retreat_plane']
SELF_COLLISIONS = False
MILLIMETER = 0.001

##################################################

def plan_sequence(robot, obstacles, assembly_network,
                  stiffness_checker=None,
                  search_method='backward', value_ordering_method='random', use_layer=True, file_name=None):
    raise NotImplementedError
    # return seq, seq_poses

class WorldPose(object):
    def __init__(self, index, value):
        self.index = index
        self.value = value
    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.index,
                                  str(np.array(point_from_pose(self.value))))

class Grasp(object):
    def __init__(self, index, num, approach, attach, retreat):
        self.index = index
        self.num = num
        self.approach = approach
        self.attach = attach
        self.retreat = retreat
    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.index, self.num)


##################################################

def load_pick_and_place(instance_name, scale=MILLIMETER):
    root_directory = os.path.dirname(os.path.abspath(__file__))
    instance_directory = os.path.join(root_directory, PICKNPLACE_DIRECTORY, instance_name)
    print('Name: {}'.format(instance_name))
    with open(os.path.join(instance_directory, 'json', PICKNPLACE_FILENAMES[instance_name]), 'r') as f:
        json_data = json.loads(f.read())

    obj_directory = os.path.join(instance_directory, 'meshes', 'collision')
    with HideOutput():
        # TODO: load static collision env
        #world = load_pybullet(os.path.join(bricks_directory, 'urdf', 'brick_demo.urdf'))
        robot = load_pybullet(os.path.join(root_directory, IRB6600_TRACK_URDF), fixed_base=True)

    # static collision
    obstacle_from_name = {}

    brick_from_index = {}
    for json_element in json_data['sequenced_elements']:
        index = json_element['order_id']
        # TODO: group objects together for one unit element
        # TODO: transform geometry based on json_element['parent_frame']
        pick_body = create_obj(os.path.join(obj_directory, json_element['element_geometry_file_names'][0]),
                               scale=scale, color=(0, 0, 1, 1))
        add_body_name(pick_body, index)

        obj_from_ee_grasp_poses = [pose_from_tform(parse_transform(json_tf)) \
                                    for json_tf in json_element['grasps']['ee_poses']]
        # pick_grasp_plane is at the top of the object with z facing downwards

        # ee_from_obj = invert(world_from_obj_pick) # Using pick frame
        pick_parent_frame = \
        pose_from_tform(parse_transform(json_element['assembly_process']['pick']['parent_frame']))
        world_from_obj_pick = \
        multiply(pick_parent_frame, pose_from_tform(parse_transform(json_element['assembly_process']['pick']['object_target_pose'])))

        place_parent_frame = \
        pose_from_tform(parse_transform(json_element['assembly_process']['place']['parent_frame']))
        world_from_obj_place = \
        multiply(place_parent_frame, pose_from_tform(parse_transform(json_element['assembly_process']['place']['object_target_pose'])))

        set_pose(pick_body, world_from_obj_place)
        # draw_pose(world_from_obj_pick, length=0.04)
        # draw_pose(world_from_obj_place, length=0.04)
        # for ee_p in obj_from_ee_grasp_poses:
        #     draw_pose(multiply(world_from_obj_pick, ee_p), length=0.04)
        #     draw_pose(multiply(world_from_obj_place, ee_p), length=0.04)

        # TODO: pick and place might have different approach tfs
        ee_from_approach_tf = pose_from_tform(parse_transform(json_element['assembly_process']['place']['grasp_from_approach_tf']))
        obj_from_ee_grasps = [Grasp(index, grasp_id, \
                                    multiply(obj_from_ee_pose, ee_from_approach_tf),
                                    obj_from_ee_pose,
                                    multiply(obj_from_ee_pose, ee_from_approach_tf),
                                     ) \
                              for grasp_id, obj_from_ee_pose in enumerate(obj_from_ee_grasp_poses)]

        # # draw approach poses
        # for ee_p in obj_from_ee_grasp_poses:
        #     draw_pose(multiply(world_from_obj_pick, ee_p, ee_from_approach_tf), length=0.04)
        #     draw_pose(multiply(world_from_obj_place, ee_p, ee_from_approach_tf), length=0.04)

        brick_from_index[index] = Brick(index=index, body=pick_body,
                                        initial_pose=world_from_obj_pick,
                                        goal_pose=world_from_obj_place,
                                        grasps=obj_from_ee_grasps,
                                        goal_supports=[]) #json_element.get('place_contact_ngh_ids', [])
        # pick_contact_ngh_ids are movable element contact partial orders
        # pick_support_surface_file_names are fixed element contact partial orders

    return robot, brick_from_index, obstacle_from_name

##################################################

def display_trajectories(assembly_network, trajectories, time_step=0.075):
    disconnect()
    if trajectories is None:
        return
    connect(use_gui=True)
    floor, robot = load_world()
    camera_base_pt = assembly_network.get_end_points(0)[0]
    camera_pt = np.array(camera_base_pt) + np.array([0.1, 0, 0.05])
    set_camera_pose(tuple(camera_pt), camera_base_pt)
    # wait_for_interrupt()

    movable_joints = get_movable_joints(robot)
    #element_bodies = dict(zip(elements, create_elements(node_points, elements)))
    #for body in element_bodies.values():
    #    set_color(body, (1, 0, 0, 0))
    # connected = set(ground_nodes)
    for trajectory in trajectories:
        #     if isinstance(trajectory, PrintTrajectory):
        #         print(trajectory, trajectory.n1 in connected, trajectory.n2 in connected,
        #               is_ground(trajectory.element, ground_nodes), len(trajectory.path))
        #         connected.add(trajectory.n2)
        #     #wait_for_interrupt()
        #     #set_color(element_bodies[element], (1, 0, 0, 1))
        last_point = None
        handles = []
        for conf in trajectory: #.path:
            set_joint_positions(robot, movable_joints, conf)
            # if isinstance(trajectory, PrintTrajectory):
            current_point = point_from_pose(get_link_pose(robot, link_from_name(robot, TOOL_FRAME)))
            if last_point is not None:
                color = (0, 0, 1) #if is_ground(trajectory.element, ground_nodes) else (1, 0, 0)
                handles.append(add_line(last_point, current_point, color=color))
            last_point = current_point
            wait_for_duration(time_step)
        # wait_for_interrupt()

    wait_for_interrupt()
    disconnect()

################################
def main(precompute=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--problem', default='toggle_rebar_cage_1', help='The name of the problem to solve')
    parser.add_argument('-v', '--viewer', action='store_true', help='Enables the viewer during planning (slow!)')
    parser.add_argument('-d', '--debug', action='store_true', help='Debug mode.')
    args = parser.parse_args()
    print('Arguments:', args)

    connect(use_gui=args.viewer)
    robot, brick_from_index, obstacle_from_name = load_pick_and_place(args.problem)

    # initial_conf = get_joint_positions(robot, get_movable_joints(robot))
    # dump_body(robot)

    # camera_pt = np.array(node_points[10]) + np.array([0.1,0,0.05])
    # target_camera_pt = node_points[0]

    # create collision bodies
    # bodies = create_elements(node_points, [tuple(e.node_ids) for e in elements])
    # for e, b in zip(elements, bodies):
    #     e.element_body = b
    #     # draw_pose(get_pose(b), length=0.004)

    # if has_gui():
    #     # pline_handle = draw_model(assembly_network, draw_tags=False)
    #     set_camera_pose(tuple(camera_pt), target_camera_pt)
    print('Continue?')
    wait_for_interrupt()
    # use_seq_existing_plan = args.parse_seq

    ####################
    # sequence planning completed
    # if has_gui():
    #     # wait_for_interrupt('Press a key to visualize the plan...')
    #     map(p.removeUserDebugItem, pline_handle)
    #     # draw_assembly_sequence(assembly_network, element_seq, seq_poses, time_step=1)

    # motion planning phase
    # assume that the robot's dof is all included in the ikfast model
    print('start sc motion planning.')

    # default sequence
    seq_assignment = list(range(len(brick_from_index)))
    element_seq = {e_id : seq_id for e_id, seq_id in zip(seq_assignment, seq_assignment)}
    with LockRenderer():
        tot_traj, graph_sizes = direct_ladder_graph_solve_picknplace(robot, brick_from_index, element_seq, obstacle_from_name)
    #     sg = SparseLadderGraph(robot, len(get_movable_joints(robot)), assembly_network, element_seq, seq_poses, obstacles)
    #     sg.find_sparse_path(max_time=2)
    #     tot_traj, graph_sizes = sg.extract_solution()
    #
    # trajectories = list(divide_list_chunks(tot_traj, graph_sizes))

    # if args.viewer:
    # display_trajectories(assembly_network, trajectories, time_step=0.15)
    print('Quit?')
    if has_gui():
        wait_for_interrupt()

if __name__ == '__main__':
    main()
