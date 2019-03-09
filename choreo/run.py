from __future__ import print_function

import cProfile
import pstats
import numpy as np
import argparse
from copy import copy

import pybullet as p

from ss-pybullet.pybullet_tools.utils import connect, disconnect, wait_for_interrupt, LockRenderer, \
    has_gui, remove_body, set_camera_pose, get_movable_joints, set_joint_positions, \
    wait_for_duration, point_from_pose, get_link_pose, link_from_name, add_line, get_collision_fn

from choreo.extrusion_utils import create_elements, \
    load_extrusion, load_world, get_disabled_collisions
from conrob_pybullet.utils.ikfast.kuka_kr6_r900.ik import TOOL_FRAME

from .assembly_datastructure import AssemblyNetwork
from .csp import CSP, backtracking_search, UniversalDict
from .csp_utils import count
from .choreo_utils import draw_model, load_end_effector, draw_assembly_sequence, PHI_DISC, THETA_DISC, \
    update_collision_map, write_seq_json, read_seq_json, cmap_id2angle, \
    check_exist_valid_kinematics, EEDirection
from .sc_cartesian_planner import divide_list_chunks, SparseLadderGraph

DEBUG = True
SELF_COLLISIONS = False

# constraint fn placeholder
def always_false_constraint_fn(A, a, B, b):
    print('dumb constraint fn called!')
    return False

class AssemblyCSP(CSP):
    """
    needs to keep track of a feasible direction map for each domain value (element_id)

    """
    def __init__(self, robot=None, obstacles=[], assembly_network=None):
        assert(isinstance(assembly_network, AssemblyNetwork))
        n = assembly_network.get_size_of_elements()
        decomposed_domains = {}
        layer_ids = assembly_network.get_layers()
        layer_ids.sort()
        for l_id in layer_ids:
            l_e_ids = assembly_network.get_layer_element_ids(l_id)
            prev_e_num = len(decomposed_domains)
            for i in range(prev_e_num, prev_e_num + len(l_e_ids)):
                decomposed_domains[i] = l_e_ids

        CSP.__init__(self, variables=list(range(n)), domains=decomposed_domains,
                     neighbors=UniversalDict(list(range(n))), constraints=always_false_constraint_fn)

        self.robot = robot
        self.disabled_collisions = get_disabled_collisions(robot)
        self.obstacles = obstacles
        self.net = assembly_network
        self.cmaps = dict()
        for e_id in range(n):
            self.cmaps[e_id] = np.ones(PHI_DISC * THETA_DISC, dtype=int)
        # self.set_layer_stage(0, [])
        self.ee_body = load_end_effector()

    def nconflicts(self, var, val, assignment):
        """Return the number of conflicts var=val has with other variables."""
        # Subclasses may implement this more efficiently
        # def conflict(var2):
        #     return (var2 in assignment and
        #             not self.constraints(var, val, var2, assignment[var2]))
        # return count(conflict(v) for v in self.neighbors[var])

        def alldiff(self, var, val, assignment):
            assignment[var] = val
            return len(assignment.values()) == len(set(assignment.values()))

        def connect(self, var, val, assignment):
            # if var not in assignment.keys():
            #     if len(assignment) is not 0: assert(var == max(assignment.keys())+1)
            ngbh_e_ids = self.net.get_element_neighbor(val)
            return any(val_k in ngbh_e_ids for val_k in assignment.values()) \
                       or self.net.is_element_grounded(val)
            # else:
            #     return True

        def stiffness(self, var, val, assignment):
            return True

        def stability(self, var, val, assignment):
            return True

        def exist_valid_ee_pose(self, var, val, assignment):
            # will printing edge #val collide with any assigned element?
            # if var in assignment.keys():
            #     return False
            # else:
            if sum(self.cmaps[val]) == 0:
                return False
            else:
                built_obstacles = self.obstacles + [self.net.get_element_body(assignment[i]) for i in assignment.keys() if i != var]
                collision_fn = get_collision_fn(self.robot, get_movable_joints(self.robot), built_obstacles,
                                        attachments=[], self_collisions=SELF_COLLISIONS,
                                        disabled_collisions=self.disabled_collisions,
                                        custom_limits={})
                val_cmap = copy(self.cmaps[val])

                # check against all existing edges except itself
                for k in assignment.keys():
                    if k == var:
                        continue
                    exist_e_id = assignment[k]
                    val_cmap = update_collision_map(self.net, self.ee_body, val, exist_e_id, val_cmap, self.obstacles,
                                                    check_ik=False)
                    if sum(val_cmap) == 0:
                        return False

                return check_exist_valid_kinematics(self.net, val, self.robot, val_cmap, collision_fn)
                # return True

        constraint_fns = [alldiff, connect, exist_valid_ee_pose]

        _nconflicts = count(not fn(self, var, val, assignment) for fn in constraint_fns)
        return _nconflicts

    def goal_test(self, state):
        """The goal is to assign all variables, with all constraints satisfied."""
        assignment = dict(state)
        return len(assignment) == len(self.variables)
                # and all(self.nconflicts(variables, assignment[variables], assignment) == 0
                #         for variables in self.variables))

    # These are for constraint propagation

    def support_pruning(self, var=None, value=None, assignment=None):
        """Make sure we can prune values from domains. (We want to pay
        for this only if we use it.)
        return a list of (var, val) tuple"""
        removals = dict({'domain':[], 'cmaps':dict()})
        if self.curr_domains is None:
            self.curr_domains = {v: list(self.domains[v]) for v in self.variables}

        assert(var is not None and value is not None and assignment is not None)

        # this is checking against the future
        unassigned_vals = list(set(range(len(self.variables))).difference(assignment.values()))
        if len(assignment.values()) == 1:
            # prune the first one against the static obstacles
            unassigned_vals.append(assignment[0])

        for u_val in unassigned_vals:
            old_cmap = copy(self.cmaps[u_val])
            self.cmaps[u_val] = \
                update_collision_map(self.net, self.ee_body, u_val, value, self.cmaps[u_val], self.obstacles, check_ik=False)
            if sum(map(abs, old_cmap - self.cmaps[u_val])) != 0:
                removals['cmaps'][u_val] = old_cmap - self.cmaps[u_val]

        return removals

    def suppose(self, var, value, assignment=None):
        """Start accumulating inferences from assuming var=value."""
        removals = self.support_pruning(var, value, assignment)

        removals['domain'].extend([(var, a) for a in self.curr_domains[var] if a != value])
        self.curr_domains[var] = [value]

        unassigned_vars = list(set(self.variables).difference(assignment.keys()))
        for u_var in unassigned_vars:
            if value in self.curr_domains[u_var]:
                self.curr_domains[u_var].remove(value) # list(set(self.domains[u_var]).difference([value]))
                removals['domain'].append((u_var, value))

        return removals

    def restore(self, removals):
        """Undo a supposition and all inferences from it."""
        # TODO
        if removals.has_key('domain'):
            for B, b in removals['domain']:
                self.curr_domains[B].append(b)

        if removals.has_key('cmaps'):
            for e_id in removals['cmaps'].keys():
                free_mask = removals['cmaps'][e_id]
                assert(len(self.cmaps[e_id]) == len(free_mask))
                self.cmaps[e_id] = self.cmaps[e_id] + free_mask

    # this is for layer decomposition (domain shrinking)
    def set_layer_stage(self, layer_id, assignment):
        """update the domain, constraint checkers based on current layer id"""
        # self.variables = range(assignment.largest var + 1, len(element_group_ids[layer_id]))
        # self.domains = for each var in variblaes, intersect(curr_domain[e], curr layer ids)

        # set constraint checker to regard previous layers' element as existing & collision
        pass

# variable ordering
def next_variable_in_sequence(assignment, csp):
    if len(assignment)>0:
        return max(assignment.keys())+1
    else:
        # randomly
        # TODO: find a grounded element
        return 0

# value ordering
# choose value with min cmap sum
def cmaps_value_ordering(var, assignment, csp):
    cur_vals =  csp.choices(var)
    return sorted(cur_vals, key=lambda val: sum(csp.cmaps[val]))

# inference
def cmaps_forward_check(csp, var, value, assignment, removals):
    """check any unprinted e's cmap sum = 0"""
    unassigned_vals = list(set(range(len(csp.variables))).difference(assignment.values()))
    for val in unassigned_vals:
        if sum(csp.cmaps[val]) == 0:
            return False
    return True

def plan_sequence(robot, obstacles, assembly_network):
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
    csp = AssemblyCSP(robot, obstacles, assembly_network=assembly_network)
    seq, csp = backtracking_search(csp, select_unassigned_variable=next_variable_in_sequence,
                        order_domain_values=cmaps_value_ordering,
                        inference=cmaps_forward_check)
    print(seq)

    pr.disable()
    pstats.Stats(pr).sort_stats('tottime').print_stats(10)

    seq_poses = {}
    for i in seq.keys():
        e_id = seq[i]
        # feasible ee directions
        seq_poses[i] = []
        assert(csp.cmaps.has_key(e_id))
        for cmap_id, free_flag in enumerate(csp.cmaps[e_id]):
            if free_flag:
                phi, theta = cmap_id2angle(cmap_id)
                seq_poses[i].append(EEDirection(phi=phi, theta=theta))

    remove_body(csp.ee_body)
    # TODO: do cmaps -> pose translation here

    return seq, seq_poses

def display_trajectories(assembly_network, trajectories, time_step=0.075):
    if trajectories is None:
        return
    connect(use_gui=True)
    floor, robot = load_world()
    camera_base_pt = assembly_network.get_end_points(0)[0]
    camera_pt = np.array(camera_base_pt) + np.array([0.1,0,0.05])
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
        #wait_for_interrupt()
    #user_input('Finish?')

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
    parser.add_argument('-s', '--parse_seq', action='store_true', help='parse sequence planning result from a file and proceed to motion planning')
    args = parser.parse_args()
    print('Arguments:', args)

    elements, node_points, ground_nodes = load_extrusion(args.problem, parse_layers=True)
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

    if has_gui():
        pline_handle = draw_model(elements, node_points, ground_nodes)
        set_camera_pose(tuple(camera_pt), target_camera_pt)
       # wait_for_interrupt('Continue?')

    # create collision bodies
    bodies = create_elements(node_points, [tuple(e.node_ids) for e in elements])
    for e, b in zip(elements, bodies):
        e.element_body = b
        # draw_pose(get_pose(b), length=0.004)

    assembly_network = AssemblyNetwork(node_points, elements, ground_nodes)

    use_seq_existing_plan = args.parse_seq
    if not use_seq_existing_plan:
        with LockRenderer():
            element_seq, seq_poses = plan_sequence(robot, obstacles, assembly_network)
        write_seq_json(assembly_network, element_seq, seq_poses, args.problem)
    else:
        element_seq, seq_poses = read_seq_json(args.problem)

    # TODO: sequence direction routing (if real fab)
    ####################
    # sequence planning completed
    if has_gui():
        # wait_for_interrupt('Press a key to visualize the plan...')
        map(p.removeUserDebugItem, pline_handle)
        draw_assembly_sequence(assembly_network, element_seq, seq_poses)

    # motion planning phase
    # assume that the robot's dof is all included in the ikfast model
    print('start sc motion planning.')

    with LockRenderer():
        # tot_traj, graph_sizes = direct_ladder_graph_solve(robot, assembly_network, element_seq, seq_poses, obstacles)
        sg = SparseLadderGraph(robot, len(get_movable_joints(robot)), assembly_network, element_seq, seq_poses, obstacles)
        sg.find_sparse_path(max_time=2)
        tot_traj, graph_sizes = sg.extract_solution()

    trajectories = list(divide_list_chunks(tot_traj, graph_sizes))

    disconnect()
    display_trajectories(assembly_network, trajectories)

    print('Quit?')
    if has_gui():
        wait_for_interrupt()
    # display_trajectories(ground_nodes, plan)

if __name__ == '__main__':
    main()
