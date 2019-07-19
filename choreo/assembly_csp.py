import numpy as np
from copy import copy
import time
import random
import datetime

from conrob_pybullet.ss_pybullet.pybullet_tools.utils import get_movable_joints, get_collision_fn

from choreo.extrusion_utils import get_disabled_collisions
from conrob_pybullet.utils.ikfast.kuka_kr6_r900.ik import TOOL_FRAME

from .assembly_datastructure import AssemblyNetwork
from .csp import CSP, UniversalDict
from .csp_utils import count
from .choreo_utils import update_collision_map, PHI_DISC, THETA_DISC, load_end_effector, check_exist_valid_kinematics, \
    update_collision_map_batch
# from pyconmech import stiffness_checker

SELF_COLLISIONS = False

# constraint fn placeholder
def always_false_constraint_fn(A, a, B, b):
    print('dumb constraint fn called!')
    return False

class AssemblyCSP(CSP):
    """
    needs to keep track of a feasible direction map for each domain value (element_id)

    """
    def __init__(self, robot=None, obstacles=[], assembly_network=None, stiffness_checker=None,
                 vom=None, search_method=None, use_layer=True):
        self.search_method = search_method

        assert(isinstance(assembly_network, AssemblyNetwork))
        n = assembly_network.get_size_of_elements()
        decomposed_domains = {}
        layer_ids = assembly_network.get_layers()

        assert(search_method and vom)
        if self.search_method == 'forward':
            sorted(layer_ids)
        if self.search_method == 'backward':
            sorted(layer_ids, reverse=True)
        self.vom = vom

        if use_layer:
            for l_id in layer_ids:
                l_e_ids = assembly_network.get_layer_element_ids(l_id)
                prev_e_num = len(decomposed_domains)
                for i in range(prev_e_num, prev_e_num + len(l_e_ids)):
                    decomposed_domains[i] = l_e_ids
        else:
            decomposed_domains = UniversalDict(list(range(n)))

        CSP.__init__(self, variables=list(range(n)), domains=decomposed_domains,
                     neighbors=UniversalDict(list(range(n))), constraints=always_false_constraint_fn)

        self.start_time = time.time()
        self.robot = robot
        self.disabled_collisions = get_disabled_collisions(robot)
        self.obstacles = obstacles
        self.net = assembly_network
        self.cmaps = dict()
        for e_id in range(n):
            self.cmaps[e_id] = np.ones(PHI_DISC * THETA_DISC, dtype=int)
        self.ee_body = load_end_effector()
        self.stiffness_checker = stiffness_checker

        self.constr_check_time = {}
        self.constr_check_time['connect'] = {}
        self.constr_check_time['connect'][1] = 0
        self.constr_check_time['connect'][0] = 0

        self.constr_check_time['exist_valid_ee_pose'] = {}
        self.constr_check_time['exist_valid_ee_pose'][1] = 0
        self.constr_check_time['exist_valid_ee_pose'][0] = 0

        self.constr_check_time['stiffness'] = {}
        self.constr_check_time['stiffness'][1] = 0
        self.constr_check_time['stiffness'][0] = 0

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

            def check_sub_graph_connect_to_ground(assembly_network, sub_e_graph):
                to_ground_dist = []
                for e in sub_e_graph:
                     to_ground_dist.append(self.net.dijkstra(e, sub_e_graph))
                min_g_dist = not any([d is np.inf for d in to_ground_dist])
                # print('ground_dist {0} / {1}'.format(min_g_dist, to_ground_dist))
                return min_g_dist

            success = False
            if self.search_method == 'forward':
                ngbh_e_ids = self.net.get_element_neighbor(val)
                success = any(val_k in ngbh_e_ids for val_k in assignment.values()) \
                          or self.net.is_element_grounded(val)
                self.constr_check_time['connect'][success] += 1
                return success

            if self.search_method == 'backward':
                unassigned_vals = list(set(range(len(self.variables))).difference(assignment.values()))
                if not unassigned_vals:
                    # all assigned
                    success = True
                    self.constr_check_time['connect'][success] += 1
                    return success

                ngbh_e_ids = self.net.get_element_neighbor(val)
                connect_to_unassigned = \
                    any(val_k in ngbh_e_ids for val_k in unassigned_vals) \
                    or all(self.net.is_element_grounded(val_k) for val_k in unassigned_vals)

                # print(' -- check e{}'.format(val))
                if any(self.net.is_element_grounded(e) for e in unassigned_vals) \
                   and connect_to_unassigned:
                    success = check_sub_graph_connect_to_ground(self.net, unassigned_vals)
                else:
                    success = False

                self.constr_check_time['connect'][success] += 1
                return success

        def stiffness(self, var, val, assignment):
            exist_e_ids = list(set(range(len(self.variables))).difference(list(assignment.values()) + [val]))
            if exist_e_ids and self.stiffness_checker:
                success = self.stiffness_checker.solve(exist_e_ids)
            else:
                success = True
            self.constr_check_time['stiffness'][success] += 1
            return success

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
                val_cmap = copy(self.cmaps[val])
                built_obstacles = self.obstacles

                success = False
                # forward search
                if self.search_method == 'forward':
                    built_obstacles = built_obstacles + [self.net.get_element_body(assignment[i]) for i in assignment.keys() if i != var]
                    # check against all existing edges except itself
                    for k in assignment.keys():
                        if k == var:
                            continue
                        exist_e_id = assignment[k]
                        # TODO: check print nodal direction n1 -> n2
                        val_cmap = update_collision_map(self.net, self.ee_body, val, exist_e_id, val_cmap, self.obstacles)
                        if sum(val_cmap) == 0:
                            success = False
                            self.constr_check_time['exist_valid_ee_pose'][success] += 1
                            return success
                # backward search
                elif self.search_method == 'backward':
                    # all unassigned values are assumed to be collision objects
                    # TODO: only check current layer's value?
                    # TODO: these set difference stuff should use domain value
                    unassigned_vals = list(set(range(len(self.variables))).difference(assignment.values()))
                    if not self.net.is_element_grounded(val):
                        ngbh_e_ids = set(self.net.get_element_neighbor(val)).intersection(unassigned_vals)
                        shared_node = set()
                        for n_e in ngbh_e_ids:
                            shared_node.update([self.net.get_shared_node_id(val, n_e)])
                        shared_node = list(shared_node)
                    else:
                        shared_node = [v_id for v_id in self.net.get_element_end_point_ids(val)
                                       if self.net.assembly_joints[v_id].is_grounded]
                    assert(shared_node)

                    # everytime we start fresh
                    # val_cmap = np.ones(PHI_DISC * THETA_DISC, dtype=int)

                    # print('checking #{0}-e{1}, before pruning, cmaps sum: {2}'.format(var, val, sum(val_cmap)))
                    # print('checking print #{} collision against: '.format(val))
                    # print(sorted(unassigned_vals))
                    # print('static obstables: {}'.format(built_obstacles))

                    built_obstacles = built_obstacles + [self.net.get_element_body(unass_val) for unass_val in unassigned_vals]
                    val_cmap = update_collision_map_batch(self.net, self.ee_body,
                                                          print_along_e_id=val, print_along_cmap=val_cmap,
                                                          printed_nodes=shared_node, bodies=built_obstacles)

                    # print('after pruning, cmaps sum: {}'.format(sum(val_cmap)))
                    # print('-----')

                    if sum(val_cmap) < 5: #== 0:
                        success = False
                        self.constr_check_time['exist_valid_ee_pose'][success] += 1
                        return success

                collision_fn = get_collision_fn(self.robot, get_movable_joints(self.robot), built_obstacles,
                                                attachments=[], self_collisions=SELF_COLLISIONS,
                                                disabled_collisions=self.disabled_collisions,
                                                custom_limits={})
                success = check_exist_valid_kinematics(self.net, val, self.robot, val_cmap, collision_fn)
                self.constr_check_time['exist_valid_ee_pose'][success] += 1
                return success

        constraint_fns = [alldiff, connect, exist_valid_ee_pose, stiffness]

        violation = [not fn(self, var, val, assignment) for fn in constraint_fns]
        # print('constraint violation: {}'.format(violation))
        _nconflicts = count(violation)
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

        if self.search_method == 'forward':
            unassigned_vals = list(set(range(len(self.variables))).difference(assignment.values()))
            if len(assignment.values()) == 1:
                # prune the first one against the static obstacles
                unassigned_vals.append(assignment[0])

            # this is checking against the unprinted elements
            # TODO: only check vals in the same layer
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

        # alldiff
        unassigned_vars = list(set(self.variables).difference(assignment.keys()))
        for u_var in unassigned_vars:
            if value in self.curr_domains[u_var]:
                self.curr_domains[u_var].remove(value)
                removals['domain'].append((u_var, value))

        return removals

    def restore(self, removals):
        """Undo a supposition and all inferences from it."""
        # TODO
        if 'domain' in removals:
            for B, b in removals['domain']:
                self.curr_domains[B].append(b)

        if 'cmaps' in removals:
            for e_id in removals['cmaps'].keys():
                free_mask = removals['cmaps'][e_id]
                assert(len(self.cmaps[e_id]) == len(free_mask))
                self.cmaps[e_id] = self.cmaps[e_id] + free_mask

    def write_csp_log(self, file_name, log_path=None):
        import os
        from collections import OrderedDict
        import json

        if not log_path:
            root_directory = os.path.dirname(os.path.abspath(__file__))
            file_dir = os.path.join(root_directory, 'csp_log')
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
        else:
            file_dir = log_path

        file_path = os.path.join(file_dir, file_name + '_'
                                 + self.search_method + '_'
                                 + self.vom + '_csp_log' + '.json')
        if not os.path.exists(file_path):
            open(file_path, "w+").close()

        data = OrderedDict()
        data['assembly_type'] = 'extrusion'
        data['file_name'] = file_name
        data['write_time'] = str(datetime.datetime.now())
        data['element_number'] = self.net.get_size_of_elements()
        data['support_number'] = self.net.get_size_of_grounded_elements()

        data['search_method'] = self.search_method
        data['value_order_method'] = self.vom
        data['number_assigns'] = self.nassigns
        data['number_backtracks'] = self.nbacktrackings
        data['solve_time_util_stop'] = time.time() - self.start_time
        data['constr_check_time'] = self.constr_check_time
        data['assign_history'] = self.log_assigns

        with open(file_path, 'w') as outfile:
            json.dump(data, outfile, indent=4)

# variable ordering
def next_variable_in_sequence(assignment, csp):
    if len(assignment)>0:
        return max(assignment.keys())+1
    else:
        return 0

# value ordering
def random_value_ordering(var, assignment, csp):
    cur_vals = csp.choices(var)
    random.shuffle(cur_vals)
    return cur_vals

# --- used in forward search
# choose value with min cmap sum
def cmaps_value_ordering(var, assignment, csp):
    cur_vals = csp.choices(var)
    return sorted(cur_vals, key=lambda val: sum(csp.cmaps[val]))

# --- used in backward search
def traversal_to_ground_value_ordering(var, assignment, csp):
    # compute graph traversal distance to the ground
    cur_vals = copy(csp.choices(var))
    return sorted(cur_vals, key=lambda val: csp.net.get_element_to_ground_dist(val), reverse=True)

# inference
# used in forward search
def cmaps_forward_check(csp, var, value, assignment, removals):
    """check any unprinted e's cmap sum = 0"""
    unassigned_vals = list(set(range(len(csp.variables))).difference(assignment.values()))
    for val in unassigned_vals:
        if sum(csp.cmaps[val]) == 0:
            return False
    return True
