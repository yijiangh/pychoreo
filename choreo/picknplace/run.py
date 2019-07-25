from __future__ import print_function
import sys, os
sys.path.append('choreo/')

import cProfile
import pstats
import numpy as np
import argparse
import time, sys
import json
from random import random

from conrob_pybullet.ss_pybullet.pybullet_tools.utils import connect, disconnect, wait_for_user, LockRenderer, \
    has_gui, remove_body, set_camera_pose, get_movable_joints, set_joint_positions, \
    wait_for_duration, point_from_pose, get_link_pose, link_from_name, add_line, user_input,\
    HideOutput, load_pybullet, create_obj, draw_pose, add_body_name, get_pose, \
    pose_from_tform, invert, multiply, set_pose, plan_joint_motion, get_joint_positions, \
    add_fixed_constraint, remove_fixed_constraint, Attachment, create_attachment, \
    pairwise_collision, set_color
from choreo.choreo_utils import draw_model, draw_assembly_sequence, write_seq_json, \
read_seq_json, cmap_id2angle, EEDirection, check_and_draw_ee_collision, \
set_cmaps_using_seq, parse_transform, extract_file_name, color_print, int2element_id, \
get_collision_fn_diagnosis
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
TRANSITION_JT_RESOLUTION = 0.001

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
    for e_id, json_element in json_data['sequenced_elements'].items():
        index = json_element['object_id']
        assert(e_id == index)
        # TODO: transform geometry based on json_element['parent_frame']

        ## parsing poses
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

        # TODO: pick and place might have different approach tfs
        ee_from_approach_tf = pose_from_tform(parse_transform(json_element['assembly_process']['place']['grasp_from_approach_tf']))
        obj_from_ee_grasps = [Grasp(index, grasp_id, \
                                    multiply(obj_from_ee_pose, ee_from_approach_tf),
                                    obj_from_ee_pose,
                                    multiply(obj_from_ee_pose, ee_from_approach_tf),
                                     ) \
                              for grasp_id, obj_from_ee_pose in enumerate(obj_from_ee_grasp_poses)]

        ## parsing geometries
        pick_bodies = []
        for sub_id, sub_dict in json_element['element_geometry_file_names'].items():
            if 'convex_decomp' in sub_dict and sub_dict['convex_decomp']:
                for part_obj_file in sub_dict['convex_decomp']:
                    pick_sub_body = create_obj(os.path.join(obj_directory, part_obj_file), scale=scale, color=(random(), random(), 1, 1))
                    pick_bodies.append(pick_sub_body)
            else:
                color_print('warning: E#{} does not have convex decomp bodies, use full body instead.'.format(index))
                pick_full_body = create_obj(os.path.join(obj_directory, sub_dict['full_obj']),
                                            scale=scale, color=(0, 0, 1, 1))
                # add_body_name(pick_full_body, index)
                pick_bodies.append(pick_full_body)

        brick_from_index[index] = Brick(index=index, body=pick_bodies,
                                        initial_pose=world_from_obj_pick,
                                        goal_pose=world_from_obj_place,
                                        grasps=obj_from_ee_grasps,
                                        goal_supports=[]) #json_element.get('place_contact_ngh_ids', [])
        # pick_contact_ngh_ids are movable element contact partial orders
        # pick_support_surface_file_names are fixed element contact partial orders

    # static collision
    obstacle_from_name = {}
    for so_name, so_dict in json_data['static_obstacles'].items():
        for sub_id, so in so_dict.items():
            # obj_name = so_name + '_' + sub_id + '_full'
            # obstacle_from_name[obj_name] =  create_obj(os.path.join(obj_directory, so['full_obj']),
            #                        scale=scale, color=(0, 0, 1, 0.4))
            # add_body_name(obstacle_from_name[obj_name], obj_name)

            for cvd_obj in so['convex_decomp']:
                obj_name = extract_file_name(cvd_obj)
                obstacle_from_name[obj_name] =  create_obj(os.path.join(obj_directory, cvd_obj), scale=scale, color=(1, random(), random(), 0.6))
                # add_body_name(obstacle_from_name[obj_name], obj_name)

    return robot, brick_from_index, obstacle_from_name


def sanity_check_collisions(brick_from_index, obstacle_from_name):
    in_collision = False
    init_pose = None
    for brick in brick_from_index.values():
        for e_body in brick.body:
            if not init_pose:
                init_pose = get_pose(e_body)
            for so_id, so in obstacle_from_name.items():
                set_pose(e_body, brick.initial_pose)
                if pairwise_collision(e_body, so):
                    set_color(e_body, (1, 0, 0, 0.6))
                    set_color(so, (1, 0, 0, 0.6))

                    in_collision = True
                    print('collision detected between brick #{} and static #{} in its pick pose'.format(brick.index, so_id))
                    wait_for_user()

                set_pose(e_body, brick.goal_pose)
                if pairwise_collision(e_body, so):
                    in_collision = True
                    print('collision detected between brick #{} and static #{} in its place pose'.format(brick.index, so_id))
                    wait_for_user()

    # # reset their poses for visual...
    for brick in brick_from_index.values():
        for e_body in brick.body:
            set_pose(e_body, init_pose)
    return in_collision
##################################################

def display_trajectories(robot, brick_from_index, element_seq, trajectories, \
                         cartesian_time_step=0.075, transition_time_step=0.1, step_sim=False):
    # enable_gravity()
    # movable_joints = get_movable_joints(robot)
    movable_joints = get_track_arm_joints(robot)
    end_effector_link = link_from_name(robot, TOOL_FRAME)

    for seq_id, unit_picknplace in enumerate(trajectories):
        handles = []
        brick = brick_from_index[element_seq[seq_id]]

        print('seq #{} : place 2 pick tranisiton'.format(seq_id))
        if unit_picknplace['place2pick']:
            # place2pick transition
            for conf in unit_picknplace['place2pick']:
                set_joint_positions(robot, movable_joints, conf)
                wait_for_duration(transition_time_step)
        else:
            print('seq #{} does not have place to pick transition plan found!'.format(seq_id))

        if step_sim: wait_for_user()

        print('seq #{} : pick approach'.format(seq_id))
        # pick_approach
        for conf in unit_picknplace['pick_approach']:
            set_joint_positions(robot, movable_joints, conf)
            wait_for_duration(cartesian_time_step)

        if step_sim: wait_for_user()

        # pick attach
        attachs = []
        for e_body in brick.body:
            attachs.append(create_attachment(robot, end_effector_link, e_body))
        # add_fixed_constraint(brick.body, robot, end_effector_link)

        print('seq #{} : pick retreat'.format(seq_id))
        # pick_retreat
        for conf in unit_picknplace['pick_retreat']:
            set_joint_positions(robot, movable_joints, conf)
            for at in attachs: at.assign()
            wait_for_duration(cartesian_time_step)

        if step_sim: wait_for_user()

        print('seq #{} : pick 2 place tranisiton'.format(seq_id))
        # pick2place transition
        if unit_picknplace['pick2place']:
            for conf in unit_picknplace['pick2place']:
                set_joint_positions(robot, movable_joints, conf)
                for at in attachs: at.assign()
                wait_for_duration(transition_time_step)
        else:
            print('seq #{} does not have pick to place transition plan found!'.format(seq_id))

        if step_sim: wait_for_user()

        print('seq #{} : place approach'.format(seq_id))
        # place_approach
        for conf in unit_picknplace['place_approach']:
            set_joint_positions(robot, movable_joints, conf)
            for at in attachs: at.assign()
            wait_for_duration(cartesian_time_step)

        if step_sim: wait_for_user()

        # place detach
        # remove_fixed_constraint(brick.body, robot, end_effector_link)

        print('seq #{} : place retreat'.format(seq_id))
        # place_retreat
        for conf in unit_picknplace['place_retreat']:
            set_joint_positions(robot, movable_joints, conf)
            wait_for_duration(cartesian_time_step)

        if step_sim: wait_for_user()

################################
def main(precompute=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--problem', default='toggle_rebar_cage_1', help='The name of the problem to solve')
    parser.add_argument('-v', '--viewer', action='store_true', help='Enables the viewer during planning (slow!)')
    # parser.add_argument('-d', '--debug', action='store_true', help='Debug mode.')
    parser.add_argument('-s', '--step_sim', action='store_true', help='stepping simulation.')
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
    print('sanity collision check passed.')

    # if has_gui():
    #     # pline_handle = draw_model(assembly_network, draw_tags=False)
    #     set_camera_pose(tuple(camera_pt), target_camera_pt)
    # print('Continue?')
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
    # element_seq[0] = 1
    # element_seq[1] = 0
    # element_seq[2] = 2

    for key, val in element_seq.items():
        element_seq[key] = int2element_id(val)

    for e_id in element_seq.values():
        # for e_body in brick_from_index[e_id].body: set_pose(e_body, brick_from_index[e_id].goal_pose)
        draw_pose(brick_from_index[e_id].initial_pose, length=0.02)
        draw_pose(brick_from_index[e_id].goal_pose, length=0.02)

    # with LockRenderer():
    tot_traj, graph_sizes = direct_ladder_graph_solve_picknplace(robot, brick_from_index, element_seq, obstacle_from_name, end_effector_link)
    #     sg = SparseLadderGraph(robot, len(get_movable_joints(robot)), assembly_network, element_seq, seq_poses, obstacles)
    #     sg.find_sparse_path(max_time=2)
    #     tot_traj, graph_sizes = sg.extract_solution()

    # print(graph_sizes)
    picknplace_cart_plans = divide_nested_list_chunks(tot_traj, graph_sizes)

    print('Cartesian planning finished.')
    wait_for_user()

    print('Transition planning starts.')
    ## transition planning
    static_obstacles = list(obstacle_from_name.values())
    # reset brick poses
    for e_id in element_seq.values():
        for e_body in brick_from_index[e_id].body:
            set_pose(e_body, brick_from_index[e_id].initial_pose)

    for seq_id, e_id in element_seq.items():
        print('transition seq#{}'.format(seq_id))
        picknplace_unit = picknplace_cart_plans[seq_id]
        # brick = brick_from_index[e_id]

        if seq_id != 0:
            tr_start_conf = picknplace_cart_plans[seq_id-1]['place_retreat'][-1]
        else:
            tr_start_conf = initial_conf
        set_joint_positions(robot, movable_joints, tr_start_conf)

        cur_mo_list = []
        for mo_id, mo in brick_from_index.items():
            if mo_id in element_seq.values():
                cur_mo_list.extend(mo.body)

        place2pick_path = plan_joint_motion(robot, movable_joints, picknplace_cart_plans[seq_id]['pick_approach'][0], obstacles=static_obstacles + cur_mo_list, self_collisions=SELF_COLLISIONS)
        if not place2pick_path:
            print('seq #{} cannot find place2pick transition'.format(seq_id))
            print('Diagnosis...')

            cfn = get_collision_fn_diagnosis(robot, movable_joints, obstacles=static_obstacles + cur_mo_list, attachments=[], self_collisions=SELF_COLLISIONS)

            print('start pose:')
            cfn(tr_start_conf)

            end_conf = picknplace_cart_plans[seq_id]['pick_approach'][0]
            print('end pose:')
            cfn(end_conf)

        # create attachement without needing to keep track of grasp...
        set_joint_positions(robot, movable_joints, picknplace_cart_plans[seq_id]['pick_retreat'][0])
        # attachs = [Attachment(robot, tool_link, invert(grasp.attach), e_body) for e_body in brick.body]
        attachs = [create_attachment(robot, end_effector_link, e_body) for e_body in brick_from_index[e_id].body]

        cur_mo_list = []
        for mo_id, mo in brick_from_index.items():
            if mo_id != e_id and mo_id in element_seq.values():
                cur_mo_list.extend(mo.body)

        set_joint_positions(robot, movable_joints, picknplace_cart_plans[seq_id]['pick_retreat'][-1])
        pick2place_path = plan_joint_motion(robot, movable_joints, picknplace_cart_plans[seq_id]['place_approach'][0], obstacles=static_obstacles + cur_mo_list, attachments=attachs, self_collisions=SELF_COLLISIONS, )
        resolutions = TRANSITION_JT_RESOLUTION*np.ones(len(movable_joints))
        if not pick2place_path:
            print('seq #{} cannot find pick2place transition'.format(seq_id))
            print('Diagnosis...')

            cfn = get_collision_fn_diagnosis(robot, movable_joints, obstacles=static_obstacles + cur_mo_list, attachments=attachs, self_collisions=SELF_COLLISIONS)

            print('start pose:')
            cfn(picknplace_cart_plans[seq_id]['pick_retreat'][-1])

            end_conf = picknplace_cart_plans[seq_id]['place_approach'][0]
            print('end pose:')
            cfn(end_conf)

        picknplace_cart_plans[seq_id]['place2pick'] = place2pick_path
        picknplace_cart_plans[seq_id]['pick2place'] = pick2place_path

        # set e_id element to goal pose
        for mo in brick_from_index[e_id].body:
            set_pose(mo, brick_from_index[e_id].goal_pose)

    print('\n*************************\nplanning completed. Simulate?')
    wait_for_user()

    if not has_gui():
        disconnect()
        connect(use_gui=True)
        robot, brick_from_index, obstacle_from_name = load_pick_and_place(args.problem)

    # reset brick poses
    for e_id in element_seq.values():
        for e_body in brick_from_index[e_id].body: set_pose(e_body, brick_from_index[e_id].initial_pose)

    display_trajectories(robot, brick_from_index, element_seq, picknplace_cart_plans, \
                         cartesian_time_step=0.075, transition_time_step=0.07, step_sim=args.step_sim)
    print('Quit?')
    if has_gui():
        wait_for_user()
        disconnect()

if __name__ == '__main__':
    main()
