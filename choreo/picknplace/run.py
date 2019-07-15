from __future__ import print_function
import sys, os
sys.path.append('choreo/')

import cProfile
import pstats
import numpy as np
import argparse
import time, sys
import json

from conrob_pybullet.ss_pybullet.pybullet_tools.utils import connect, disconnect, wait_for_user, LockRenderer, \
    has_gui, remove_body, set_camera_pose, get_movable_joints, set_joint_positions, \
    wait_for_duration, point_from_pose, get_link_pose, link_from_name, add_line, user_input,\
    HideOutput, load_pybullet, create_obj, draw_pose, add_body_name, get_pose, \
    pose_from_tform, invert, multiply, set_pose, plan_joint_motion, get_joint_positions, \
    add_fixed_constraint, remove_fixed_constraint, Attachment, create_attachment, \
    pairwise_collision, set_color
from choreo.choreo_utils import draw_model, draw_assembly_sequence, write_seq_json, \
read_seq_json, cmap_id2angle, EEDirection, check_and_draw_ee_collision, \
set_cmaps_using_seq, parse_transform
from choreo.sc_cartesian_planner import divide_nested_list_chunks, SparseLadderGraph, direct_ladder_graph_solve_picknplace
from choreo.assembly_datastructure import Brick
try:
    # from conrob_pybullet.utils.ikfast.kuka_kr6_r900.ik import sample_tool_ik, TOOL_FRAME
    from conrob_pybullet.utils.ikfast.abb_irb6600_track.ik import sample_tool_ik, TOOL_FRAME, get_track_arm_joints
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
IRB6600_TRACK_URDF = os.path.join('..','..','conrob_pybullet','models','abb_irb6600_track','urdf','irb6600_track.urdf')
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
        self.index = index # brick index
        self.num = num # grasp id
        self.approach = approach
        self.attach = attach
        self.retreat = retreat
    def __repr__(self):
        return '{}(b {}, g {})'.format(self.__class__.__name__, self.index, self.num)


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

        # set_pose(pick_body, world_from_obj_place)
        # draw_pose(world_from_obj_pick, length=0.04)
        # draw_pose(world_from_obj_place, length=0.04)

        # print('---{0}'.format(index))
        # print(multiply(world_from_obj_pick, obj_from_ee_grasp_poses[0]))
        # draw_pose(multiply(world_from_obj_pick, obj_from_ee_grasp_poses[0]), length=0.04)
        # print(multiply(world_from_obj_place, obj_from_ee_grasp_poses[0]))
        # draw_pose(multiply(world_from_obj_place, obj_from_ee_grasp_poses[0]), length=0.04)

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

        # draw approach poses
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

    # static collision
    obstacle_from_name = {}
    print(json_data['static_obstacles'])
    for so_list in json_data['static_obstacles'].values():
        for so in so_list:
            obstacle_from_name[so] = create_obj(os.path.join(obj_directory, so),
                                   scale=scale, color=(0, 1, 0, 0.6))
            # add_body_name(obstacle_from_name[so], so)

    return robot, brick_from_index, obstacle_from_name


def sanity_check_collisions(brick_from_index, obstacle_from_name):
    in_collision = False
    for brick in brick_from_index.values():
        for so_id, so in obstacle_from_name.items():
            set_pose(brick.body, brick.initial_pose)
            if pairwise_collision(brick.body, so):
                set_color(brick.body, (1, 0, 0, 0.6))
                set_color(so, (1, 0, 0, 0.6))

                in_collision = True
                print('collision detected between brick #{} and static #{} in its pick pose'.format(brick.index, so_id))
                wait_for_user()

            set_pose(brick.body, brick.goal_pose)
            if pairwise_collision(brick.body, so):
                in_collision = True
                print('collision detected between brick #{} and static #{} in its place pose'.format(brick.index, so_id))
                wait_for_user()

    return in_collision
##################################################

def display_trajectories(robot, brick_from_index, element_seq, trajectories, time_step=0.075):
    # enable_gravity()
    # movable_joints = get_movable_joints(robot)
    movable_joints = get_track_arm_joints(robot)
    end_effector_link = link_from_name(robot, TOOL_FRAME)

    for seq_id, unit_picknplace in enumerate(trajectories):
        handles = []
        brick = brick_from_index[element_seq[seq_id]]

        # place2pick transition
        for conf in unit_picknplace['place2pick']:
            set_joint_positions(robot, movable_joints, conf)
            wait_for_duration(time_step)

        # pick_approach
        for conf in unit_picknplace['pick_approach']:
            set_joint_positions(robot, movable_joints, conf)
            wait_for_duration(time_step)

        # pick attach
        attach = create_attachment(robot, end_effector_link, brick.body)
        # add_fixed_constraint(brick.body, robot, end_effector_link)

        # pick_retreat
        for conf in unit_picknplace['pick_retreat']:
            set_joint_positions(robot, movable_joints, conf)
            attach.assign()
            wait_for_duration(time_step)

        # pick2place transition
        for conf in unit_picknplace['pick2place']:
            set_joint_positions(robot, movable_joints, conf)
            attach.assign()
            wait_for_duration(time_step)

        # place_approach
        for conf in unit_picknplace['place_approach']:
            set_joint_positions(robot, movable_joints, conf)
            attach.assign()
            wait_for_duration(time_step)

        # place detach
        # remove_fixed_constraint(brick.body, robot, end_effector_link)

        # place_retreat
        for conf in unit_picknplace['place_retreat']:
            set_joint_positions(robot, movable_joints, conf)
            wait_for_duration(time_step)

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
    # draw the base frame
    draw_pose(pose_from_tform(parse_transform(np.eye(4))))

    movable_joints = get_track_arm_joints(robot)
    end_effector_link = link_from_name(robot, TOOL_FRAME)
    initial_conf = get_joint_positions(robot, movable_joints)
    # dump_body(robot)

    # camera_pt = np.array(node_points[10]) + np.array([0.1,0,0.05])
    # target_camera_pt = node_points[0]

    sanity_check_collisions(brick_from_index, obstacle_from_name)

    # if has_gui():
    #     # pline_handle = draw_model(assembly_network, draw_tags=False)
    #     set_camera_pose(tuple(camera_pt), target_camera_pt)
    print('Continue?')
    # use_seq_existing_plan = args.parse_seq

    ####################
    # sequence planning completed
    # if has_gui():
        # wait_for_interrupt('Press a key to visualize the plan...')
        # map(p.removeUserDebugItem, pline_handle)
        # draw_assembly_sequence(assembly_network, element_seq, seq_poses, time_step=1)

    # motion planning phase
    # assume that the robot's dof is all included in the ikfast model
    print('start sc motion planning.')

    # default sequence
    from random import shuffle
    seq_assignment = list(range(len(brick_from_index)))
    # shuffle(seq_assignment)
    element_seq = {seq_id : e_id for seq_id, e_id in enumerate(seq_assignment)}
    # element_seq = {}
    # element_seq[0] = 0
    # element_seq[1] = 1
    # element_seq[2] = 2

    for e_id in element_seq.values():
        set_pose(brick_from_index[e_id].body, brick_from_index[e_id].goal_pose)
        draw_pose(brick_from_index[e_id].initial_pose, length=0.02)
        draw_pose(brick_from_index[e_id].goal_pose, length=0.02)

    # with LockRenderer():
    tot_traj, graph_sizes = direct_ladder_graph_solve_picknplace(robot, brick_from_index, element_seq, obstacle_from_name, end_effector_link)
    #     sg = SparseLadderGraph(robot, len(get_movable_joints(robot)), assembly_network, element_seq, seq_poses, obstacles)
    #     sg.find_sparse_path(max_time=2)
    #     tot_traj, graph_sizes = sg.extract_solution()

    print(graph_sizes)
    picknplace_cart_plans = divide_nested_list_chunks(tot_traj, graph_sizes)

    wait_for_user()

    ## transition planning
    moving_obstacles = {}
    static_obstacles = list(obstacle_from_name.values())
    # reset brick poses
    for e_id in element_seq.values():
        set_pose(brick_from_index[e_id].body, brick_from_index[e_id].initial_pose)
        moving_obstacles[e_id] = brick_from_index[e_id].body

    for seq_id, e_id in element_seq.items():
        picknplace_unit = picknplace_cart_plans[seq_id]
        brick = brick_from_index[e_id]

        if seq_id != 0:
            set_joint_positions(robot, movable_joints, picknplace_cart_plans[seq_id-1]['place_retreat'][-1])
        else:
            set_joint_positions(robot, movable_joints, initial_conf)
        place2pick_path = plan_joint_motion(robot, movable_joints, picknplace_cart_plans[seq_id]['pick_approach'][0], obstacles=static_obstacles + list(moving_obstacles.values()), self_collisions=SELF_COLLISIONS)

        set_joint_positions(robot, movable_joints, picknplace_cart_plans[seq_id-1]['pick_retreat'][0])
        attachs = [create_attachment(robot, end_effector_link, brick.body)]

        tmp = moving_obstacles[e_id]
        del moving_obstacles[e_id]
        set_joint_positions(robot, movable_joints, picknplace_cart_plans[seq_id-1]['pick_retreat'][-1])
        pick2place_path = plan_joint_motion(robot, movable_joints, picknplace_cart_plans[seq_id]['place_approach'][0], obstacles=static_obstacles + list(moving_obstacles.values()), attachments=attachs, self_collisions=SELF_COLLISIONS)

        picknplace_cart_plans[seq_id]['place2pick'] = place2pick_path
        picknplace_cart_plans[seq_id]['pick2place'] = pick2place_path

        moving_obstacles[e_id] = tmp
        set_pose(moving_obstacles[e_id], brick_from_index[e_id].goal_pose)

    # reset objects to initial poses
    for e_id in element_seq.values():
        set_pose(brick_from_index[e_id].body, brick_from_index[e_id].initial_pose)
    print('planning completed. Simulate?')
    wait_for_user()

    display_trajectories(robot, brick_from_index, element_seq, picknplace_cart_plans, time_step=0.02)
    print('Quit?')
    if has_gui():
        wait_for_user()
        disconnect()

if __name__ == '__main__':
    main()
