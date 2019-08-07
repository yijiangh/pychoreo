from __future__ import print_function

import cProfile
import pstats
import numpy as np
import argparse
import time, sys
from webbrowser import open_new_tab

import pybullet as p

from conrob_pybullet.ss_pybullet.pybullet_tools.utils import connect, disconnect, wait_for_user, LockRenderer, \
    has_gui, remove_body, set_camera_pose, get_movable_joints, set_joint_positions, \
    wait_for_duration, point_from_pose, get_link_pose, link_from_name, add_line, \
    plan_joint_motion, get_joint_positions, remove_all_debug, get_name

from choreo.extrusion.extrusion_utils import create_elements, \
    load_extrusion, load_world, get_disabled_collisions
from conrob_pybullet.utils.ikfast.kuka_kr6_r900.ik import TOOL_FRAME

from pyconmech import stiffness_checker

from choreo.assembly_datastructure import AssemblyNetwork
from choreo.csp import backtracking_search
from choreo.assembly_csp import AssemblyCSP, next_variable_in_sequence, cmaps_value_ordering, cmaps_forward_check, \
    traversal_to_ground_value_ordering, random_value_ordering
from choreo.choreo_utils import draw_model, draw_assembly_sequence, write_seq_json, read_seq_json, cmap_id2angle, EEDirection, \
    check_and_draw_ee_collision, set_cmaps_using_seq, get_collision_fn_diagnosis
from choreo.sc_cartesian_planner import divide_list_chunks, SparseLadderGraph, SELF_COLLISIONS

try:
    import meshcat
    from choreo.result_viz import meshcat_visualize_assembly_sequence
except ImportError as e:
    print('\x1b[6;30;43m' + '{}, meshcat module not imported.'.format(e) + '\x1b[0m')
    USE_MESHCAT = False
    user_input("Press Enter to continue...")
else:
    USE_MESHCAT = True
USE_MESHCAT = False

__all__ = [
    'AssemblyCSP',
]

SPARSE_LADDER_GRAPH_SOLVE_TIMEOUT = 2

LOG_CSP = True
LOG_CSP_PATH = None
#"/Users/yijiangh/Dropbox (MIT)/Projects/Choreo/Software/shared_problem_instance/csp_log"

SEARCH_METHODS = {
    'b': 'backward',
    'f': 'forward',
}

def plan_sequence(robot, obstacles, assembly_network,
                  stiffness_checker=None,
                  search_method='backward', value_ordering_method='random', use_layer=True, file_name=None):
    pr = cProfile.Profile()
    pr.enable()

    # elements = dict(zip([e.e_id for e in elements_list], elements_list))
    # # partition elements into groups
    # element_group_ids = dict()
    # for e in elements_list:
    #     if e.layer_id in element_group_ids.keys():
    #         element_group_ids[e.layer_id].append(e.e_id)
    #     else:
    #         element_group_ids[e.layer_id] = []
    # layer_size = max(element_group_ids.keys())

    # generate AssemblyCSP problem
    print('search method: {0},\nvalue ordering method: {1},\nuse_layer: {2}'.format(
        search_method, value_ordering_method, use_layer))
    csp = AssemblyCSP(robot, obstacles, assembly_network=assembly_network,
                      stiffness_checker=stiffness_checker,
                      search_method=search_method,
                      vom=value_ordering_method,
                      use_layer=use_layer)
    csp.logging = LOG_CSP

    try:
        if search_method == 'forward':
            if value_ordering_method == 'random':
                seq, csp = backtracking_search(csp, select_unassigned_variable=next_variable_in_sequence,
                                               order_domain_values=random_value_ordering,
                                               inference=cmaps_forward_check)
            else:
                seq, csp = backtracking_search(csp, select_unassigned_variable=next_variable_in_sequence,
                                               order_domain_values=cmaps_value_ordering,
                                               inference=cmaps_forward_check)
        elif search_method == 'backward':
            if value_ordering_method == 'random':
                seq, csp = backtracking_search(csp, select_unassigned_variable=next_variable_in_sequence,
                                               order_domain_values=random_value_ordering)
            else:
                seq, csp = backtracking_search(csp, select_unassigned_variable=next_variable_in_sequence,
                                               order_domain_values=traversal_to_ground_value_ordering)
    except KeyboardInterrupt:
        if csp.logging and file_name:
            csp.write_csp_log(file_name, log_path=LOG_CSP_PATH)

        pr.disable()
        pstats.Stats(pr).sort_stats('tottime').print_stats(10)
        sys.exit()

    pr.disable()
    pstats.Stats(pr).sort_stats('tottime').print_stats(10)

    print('# of assigns: {0}'.format(csp.nassigns))
    print('# of bt: {0}'.format(csp.nbacktrackings))
    print('constr check time: {0}'.format(csp.constr_check_time))
    # print('final seq: {}'.format(seq))

    if search_method == 'backward':
        order_keys = list(seq.keys())
        order_keys.reverse()
        rev_seq = {}
        for i in seq.keys():
            rev_seq[order_keys[i]] = seq[i]
        seq = rev_seq
        st_time = time.time()
        csp = set_cmaps_using_seq(rev_seq, csp)
        print('pruning time: {0}'.format(time.time() - st_time))

    seq_poses = {}
    for i in sorted(seq.keys()):
        e_id = seq[i]
        # feasible ee directions
        seq_poses[i] = []
        assert(e_id in csp.cmaps)
        for cmap_id, free_flag in enumerate(csp.cmaps[e_id]):
            if free_flag:
                phi, theta = cmap_id2angle(cmap_id)
                seq_poses[i].append(EEDirection(phi=phi, theta=theta))

    remove_body(csp.ee_body)
    if csp.logging and file_name:
        csp.write_csp_log(file_name, log_path=LOG_CSP_PATH)

    return seq, seq_poses

def display_trajectories(assembly_network, process_trajs, extrusion_time_step=0.075, transition_time_step=0.1):
    disconnect()
    assert(process_trajs)

    connect(use_gui=True)
    floor, robot = load_world()
    camera_base_pt = assembly_network.get_end_points(0)[0]
    camera_pt = np.array(camera_base_pt) + np.array([0.1, 0, 0.05])
    set_camera_pose(tuple(camera_pt), camera_base_pt)

    movable_joints = get_movable_joints(robot)
    #element_bodies = dict(zip(elements, create_elements(node_points, elements)))
    #for body in element_bodies.values():
    #    set_color(body, (1, 0, 0, 0))
    # connected = set(ground_nodes)
    for seq_id, unit_proc in sorted(process_trajs.items()):
        #     if isinstance(trajectory, PrintTrajectory):
        #         print(trajectory, trajectory.n1 in connected, trajectory.n2 in connected,
        #               is_ground(trajectory.element, ground_nodes), len(trajectory.path))
        #         connected.add(trajectory.n2)
        #     #wait_for_interrupt()
        #     #set_color(element_bodies[element], (1, 0, 0, 1))
        last_point = None
        handles = []

        if 'transition' in unit_proc and unit_proc['transition']:
            for conf in unit_proc['transition']:
                set_joint_positions(robot, movable_joints, conf)
                wait_for_duration(transition_time_step)
        else:
            print('seq #{} does not have transition traj found!'.format(seq_id))

        for conf in unit_proc['print']:
            set_joint_positions(robot, movable_joints, conf)
            # if isinstance(trajectory, PrintTrajectory):
            current_point = point_from_pose(get_link_pose(robot, link_from_name(robot, TOOL_FRAME)))
            if last_point is not None:
                color = (0, 0, 1) #if is_ground(trajectory.element, ground_nodes) else (1, 0, 0)
                handles.append(add_line(last_point, current_point, color=color))
            last_point = current_point
            wait_for_duration(extrusion_time_step)

    print('simulation done.')
    wait_for_user()
    disconnect()

################################
def main(precompute=False):
    parser = argparse.ArgumentParser()
    # four-frame | simple_frame | djmm_test_block | mars_bubble | sig_artopt-bunny | topopt-100 | topopt-205 | topopt-310 | voronoi | dented_cube
    parser.add_argument('-p', '--problem', default='simple_frame', help='The name of the problem to solve')
    parser.add_argument('-sm', '--search_method', default='b', help='csp search method, b for backward, f for forward.')
    parser.add_argument('-vom', '--value_order_method', default='sp',
                        help='value ordering method, sp for special heuristic, random for random value ordering')
    parser.add_argument('-l', '--use_layer', action='store_true', help='use layer info in the search.')
    # parser.add_argument('-c', '--cfree', action='store_true', help='Disables collisions with obstacles')
    parser.add_argument('-m', '--motions', action='store_true', help='Plans motions between each extrusion')
    parser.add_argument('-v', '--viewer', action='store_true', help='Enables the viewer during planning (slow!)')
    parser.add_argument('-s', '--parse_seq', action='store_true', help='parse sequence planning result from a file and proceed to motion planning')
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

    # TODO: import other static obstacles
    static_obstacles = [floor]
    movable_joints = get_movable_joints(robot)
    disabled_collisions = get_disabled_collisions(robot)
    initial_conf = get_joint_positions(robot, movable_joints)

    camera_pt = np.array(node_points[10]) + np.array([0.1,0,0.05])
    target_camera_pt = node_points[0]

    # create collision bodies
    bodies = create_elements(node_points, [tuple(e.node_ids) for e in elements])
    for e, b in zip(elements, bodies):
        e.element_body = b
        # draw_pose(get_pose(b), length=0.004)

    assembly_network = AssemblyNetwork(node_points, elements, ground_nodes)
    assembly_network.compute_traversal_to_ground_dist()

    sc = stiffness_checker(json_file_path=file_path, verbose=False)
    # sc.set_self_weight_load(True)
    print('test stiffness check on the whole structure: {0}'.format(sc.solve()))

    if has_gui():
        pline_handle = draw_model(assembly_network, draw_tags=False)
        set_camera_pose(tuple(camera_pt), target_camera_pt)
        wait_for_user()

    use_seq_existing_plan = args.parse_seq
    if not use_seq_existing_plan:
        with LockRenderer():
            search_method = SEARCH_METHODS[args.search_method]
            element_seq, seq_poses = plan_sequence(robot, static_obstacles, assembly_network,
                                                   stiffness_checker=sc,
                                                   search_method=search_method,
                                                   value_ordering_method=args.value_order_method,
                                                   use_layer=args.use_layer,
                                                   file_name=args.problem)
        write_seq_json(assembly_network, element_seq, seq_poses, args.problem)
    else:
        element_seq, seq_poses = read_seq_json(args.problem)

    # TODO: sequence direction routing (if real fab)
    ####################
    # sequence planning completed
    if has_gui():
        # wait_for_interrupt('Press a key to visualize the plan...')
        # map(p.removeUserDebugItem, pline_handle)
        remove_all_debug()
        # draw_assembly_sequence(assembly_network, element_seq, seq_poses, time_step=1)

    if USE_MESHCAT:
        print('Visualizing assembly seq in meshcat...')
        vis = meshcat.Visualizer()
        try:
            vis.open()
        except:
            vis.url()
        meshcat_visualize_assembly_sequence(vis, assembly_network, element_seq, seq_poses, scale=3, time_step=0.5, direction_len=0.025)

    # motion planning phase
    # assume that the robot's dof is all included in the ikfast model
    print('start sc motion planning.')
    with LockRenderer():
        # tot_traj, graph_sizes = direct_ladder_graph_solve(robot, assembly_network, element_seq, seq_poses, static_obstacles)
        sg = SparseLadderGraph(robot, len(get_movable_joints(robot)), assembly_network, element_seq, seq_poses, static_obstacles)
        sg.find_sparse_path(max_time=SPARSE_LADDER_GRAPH_SOLVE_TIMEOUT)
        tot_traj, graph_sizes = sg.extract_solution()

    process_trajs = {}
    for seq_id, print_jt_list in enumerate(list(divide_list_chunks(tot_traj, graph_sizes))):
        process_trajs[seq_id] = {}
        process_trajs[seq_id]['print'] = print_jt_list

    # TODO: a separate function
    # transition planning
    if args.motions:
        print('start transition planning.')
        moving_obstacles = {}
        for seq_id, e_id in sorted(element_seq.items()):
            print('transition planning # {} - E#{}'.format(seq_id, e_id))
            # print('moving obs: {}'.format([get_name(mo) for mo in moving_obstacles.values()]))
            # print('static obs: {}'.format([get_name(so) for so in static_obstacles]))
            print('---')

            if seq_id != 0:
                set_joint_positions(robot, movable_joints, process_trajs[seq_id-1]['print'][-1])
            else:
                set_joint_positions(robot, movable_joints, initial_conf)

            transition_traj = plan_joint_motion(robot, movable_joints, process_trajs[seq_id]['print'][0], obstacles=static_obstacles + list(moving_obstacles.values()), self_collisions=SELF_COLLISIONS)

            if not transition_traj:
                add_line(*assembly_network.get_end_points(e_id))

                cfn = get_collision_fn_diagnosis(robot, movable_joints, obstacles=static_obstacles + list(moving_obstacles.values()), attachments=[], self_collisions=SELF_COLLISIONS, disabled_collisions=disabled_collisions)

                st_conf = get_joint_positions(robot, movable_joints)
                print('start extrusion pose:')
                cfn(st_conf)

                end_conf = process_trajs[seq_id]['print'][0]
                print('end extrusion pose:')
                cfn(end_conf)

            process_trajs[seq_id]['transition'] = transition_traj

            e_body = assembly_network.get_element_body(e_id)
            moving_obstacles[seq_id] = e_body

        print('transition planning done! proceed to simulation?')
        wait_for_user()

    display_trajectories(assembly_network, process_trajs, extrusion_time_step=0.15, transition_time_step=0.1)
    print('Quit?')
    if has_gui():
        wait_for_user()

if __name__ == '__main__':
    main()
