from __future__ import print_function

import sys
sys.path.append('choreo/')

import cProfile
import pstats
import numpy as np
import argparse
import time, sys

import pybullet as p

from conrob_pybullet.ss_pybullet.pybullet_tools.utils import connect, disconnect, wait_for_interrupt, LockRenderer, \
    has_gui, remove_body, set_camera_pose, get_movable_joints, set_joint_positions, \
    wait_for_duration, point_from_pose, get_link_pose, link_from_name, add_line

from choreo.extrusion_utils import create_elements, \
    load_extrusion, load_world
from conrob_pybullet.utils.ikfast.kuka_kr6_r900.ik import TOOL_FRAME

# from pyconmech import stiffness_checker

from choreo.choreo_utils import draw_model, draw_assembly_sequence, write_seq_json, read_seq_json, cmap_id2angle, EEDirection, \
    check_and_draw_ee_collision, set_cmaps_using_seq
from choreo.sc_cartesian_planner import divide_list_chunks, SparseLadderGraph

import meshcat
from choreo.result_viz import meshcat_visualize_assembly_sequence


def plan_sequence(robot, obstacles, assembly_network,
                  stiffness_checker=None,
                  search_method='backward', value_ordering_method='random', use_layer=True, file_name=None):
    raise NotImplementedError
    # return seq, seq_poses

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
    # four-frame | simple_frame | djmm_test_block | mars_bubble | sig_artopt-bunny | topopt-100 | topopt-205 | topopt-310 | voronoi
    parser.add_argument('-p', '--problem', default='simple_frame', help='The name of the problem to solve')
    # parser.add_argument('-c', '--cfree', action='store_true', help='Disables collisions with obstacles')
    # parser.add_argument('-m', '--motions', action='store_true', help='Plans motions between each extrusion')
    parser.add_argument('-v', '--viewer', action='store_true', help='Enables the viewer during planning (slow!)')
    # parser.add_argument('-s', '--parse_seq', action='store_true', help='parse sequence planning result from a file and proceed to motion planning')
    parser.add_argument('-d', '--debug', action='store_true', help='Debug mode.')
    args = parser.parse_args()
    print('Arguments:', args)

    elements, node_points, ground_nodes, file_path = load_extrusion(args.problem, parse_layers=True)
    node_order = list(range(len(node_points)))

    # vert indices sanity check
    ground_nodes = [n for n in ground_nodes if n in node_order]
    elements = [element for element in elements if all(n in node_order for n in element.node_ids)]

    connect(use_gui=args.viewer)
    floor, robot = load_world()
    # static obstacles
    obstacles = [floor]
    # initial_conf = get_joint_positions(robot, get_movable_joints(robot))
    # dump_body(robot)

    camera_pt = np.array(node_points[10]) + np.array([0.1,0,0.05])
    target_camera_pt = node_points[0]

    # create collision bodies
    bodies = create_elements(node_points, [tuple(e.node_ids) for e in elements])
    for e, b in zip(elements, bodies):
        e.element_body = b
        # draw_pose(get_pose(b), length=0.004)

    if has_gui():
        pline_handle = draw_model(assembly_network, draw_tags=False)
        set_camera_pose(tuple(camera_pt), target_camera_pt)
        wait_for_interrupt('Continue?')

    use_seq_existing_plan = args.parse_seq

    ####################
    # sequence planning completed
    if has_gui():
        # wait_for_interrupt('Press a key to visualize the plan...')
        map(p.removeUserDebugItem, pline_handle)
        # draw_assembly_sequence(assembly_network, element_seq, seq_poses, time_step=1)

    # print('Visualizing assembly seq in meshcat...')
    # vis = meshcat.Visualizer()
    # try:
    #     vis.open()
    # except:
    #     vis.url()
    # meshcat_visualize_assembly_sequence(vis, assembly_network, element_seq, seq_poses,
    #                                     scale=3, time_step=0.5, direction_len=0.025)

    # motion planning phase
    # assume that the robot's dof is all included in the ikfast model
    print('start sc motion planning.')

    with LockRenderer():
        # tot_traj, graph_sizes = direct_ladder_graph_solve(robot, assembly_network, element_seq, seq_poses, obstacles)
        sg = SparseLadderGraph(robot, len(get_movable_joints(robot)), assembly_network, element_seq, seq_poses, obstacles)
        sg.find_sparse_path(max_time=2)
        tot_traj, graph_sizes = sg.extract_solution()

    trajectories = list(divide_list_chunks(tot_traj, graph_sizes))

    # if args.viewer:
    # display_trajectories(assembly_network, trajectories, time_step=0.15)
    print('Quit?')
    if has_gui():
        wait_for_interrupt()

if __name__ == '__main__':
    main()
