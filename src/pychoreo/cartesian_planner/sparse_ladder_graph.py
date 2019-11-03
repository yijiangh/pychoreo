import numpy as np
from numpy.linalg import norm
import time
import random
from copy import deepcopy, copy
from itertools import product

import pybullet as pyb
from choreo.choreo_utils import WAYPOINT_DISC_LEN, interpolate_straight_line_pts, get_collision_fn, \
 generate_way_point_poses, make_print_pose, sample_ee_yaw, interpolate_cartesian_poses, snap_sols, \
 interpolate_poses_by_num
# from conrob_pybullet.utils.ikfast.kuka_kr6_r900.ik import sample_tool_ik
# from conrob_pybullet.utils.ikfast.abb_irb6600_track.ik import sample_tool_ik, get_track_arm_joints, get_track_joint

from choreo.assembly_datastructure import AssemblyNetwork
from conrob_pybullet.ss_pybullet.pybullet_tools.utils import Pose, \
    get_movable_joints, multiply, Attachment, set_pose, invert, draw_pose, wait_for_interrupt, set_joint_positions, \
    wait_for_user, remove_debug, remove_body, has_gui, joints_from_names, link_from_name, matrix_from_quat, \
    get_joint_limits, interpolate_poses, wait_for_duration
from choreo.extrusion.extrusion_utils import get_disabled_collisions

# this is temporal...
from compas_fab.backends.ros.plugins_choreo import sample_tool_ik, best_sol

DEBUG=False
DEFAULT_UNIT_PROCESS_TIMEOUT = 10
DEFAULT_SPARSE_GRAPH_RUNG_TIMEOUT = 4
SELF_COLLISIONS=True

#########################################
# sparse ladder graph

class CapVert(object):
    def __init__(self, dof, rung_id=None):
        self.dof = dof
        self.rung_id = None
        self.z_axis_angle = None
        self.quat_pose = None
        self.st_jt_data = []
        self.end_jt_data = []
        self.__to_parent_cost = np.inf
        self.__parent_vert = None

    def distance_to(self, v):
        if not v:
            return 0
        assert(isinstance(v, CapVert))
        assert(self.dof == v.dof)
        cost = np.inf
        dof = self.dof
        n_prev_end = int(len(v.end_jt_data) / dof)
        n_this_st = int(len(self.st_jt_data) / dof)

        for i in range(n_prev_end):
            prev_end_id = i * dof
            for j in range(n_this_st):
                this_st_id = j * dof
                delta_buffer = []
                # TODO: assign weight on joints
                for k in range(dof):
                    delta_buffer.append(abs(v.end_jt_data[prev_end_id + k] - self.st_jt_data[this_st_id + k]))
                tmp_cost = sum(delta_buffer)
                if tmp_cost < cost:
                    cost = tmp_cost
        return cost

    @property
    def parent_cost(self):
        return self.__to_parent_cost

    @parent_cost.setter
    def parent_cost(self, c):
        self.__to_parent_cost = c

    @property
    def parent_vert(self):
        return self.__parent_vert

    @parent_vert.setter
    def parent_vert(self, v):
        assert(isinstance(v, CapVert) or not v)
        self.__parent_vert = v
        self.parent_cost = self.distance_to(v)

    def get_cost_to_root(self):
        prev_v = self.parent_vert
        cost = self.parent_cost
        while prev_v:
            if prev_v.parent_vert:
                cost += prev_v.parent_cost
            prev_v = prev_v.parent_vert
        return cost

class CapRung(object):
    def __init__(self):
        self.cap_verts = []
        self.path_pts = []
        # all the candidate orientations for each kinematics segment
        self.ee_dirs = []
        self.obstacles = [] # TODO: store index, point to a shared list of obstacles
        self.collision_fn = None

def generate_sample(cap_rung, cap_vert):
    assert(cap_vert)
    assert(isinstance(cap_rung, CapRung) and isinstance(cap_vert, CapVert))
    dir_sample = random.choice(cap_rung.ee_dirs)
    ee_yaw = sample_ee_yaw()

    poses = []
    cap_vert.z_axis_angle = ee_yaw
    cap_vert.quat_pose = make_print_pose(dir_sample[0], dir_sample[1], ee_yaw)
    for pt in cap_rung.path_pts:
        poses.append(multiply(Pose(point=pt), make_print_pose(dir_sample.phi, dir_sample.theta, ee_yaw)))

    return poses, cap_vert

def check_cap_vert_feasibility(robot, poses, cap_rung, cap_vert):
    assert(isinstance(cap_rung, CapRung) and isinstance(cap_vert, CapVert))
    assert(cap_rung.path_pts)
    assert(len(cap_rung.path_pts) == len(poses))

    # check ik feasibility for each of the path points
    for i, pose in enumerate(poses):
        jt_list = sample_tool_ik(robot, pose, get_all=True)
        jt_list = [jts for jts in jt_list if jts and not cap_rung.collision_fn(jts)]

        if not jt_list or all(not jts for jts in jt_list):
            # print('#{0}/{1} path pt break'.format(i,len(poses)))
            return None
        else:
            # only store the first and last sol
            if i == 0:
               cap_vert.st_jt_data = [jt for jt_l in jt_list for jt in jt_l]
            if i == len(poses)-1:
               cap_vert.end_jt_data = [jt for jt_l in jt_list for jt in jt_l]

    return cap_vert

class SparseLadderGraph(object):
    def __init__(self, robot, dof, assembly_network, element_seq, seq_poses, static_obstacles=[]):
        """seq_poses = {e_id: [(phi, theta)], ..}"""
        assert(isinstance(assembly_network, AssemblyNetwork))
        assert(isinstance(dof, int))
        self.dof = dof
        self.cap_rungs = []
        self.robot = robot
        # TODO: this shouldn't be here
        disabled_collisions = get_disabled_collisions(robot)
        built_obstacles = copy(static_obstacles)

        seq = set()
        for i in sorted(element_seq.keys()):
            e_id = element_seq[i]
            # TODO: temporal fix, this should be consistent with the seq search!!!
            if not assembly_network.is_element_grounded(e_id):
                ngbh_e_ids = seq.intersection(assembly_network.get_element_neighbor(e_id))
                shared_node = set()
                for n_e in ngbh_e_ids:
                    shared_node.update([assembly_network.get_shared_node_id(e_id, n_e)])
                shared_node = list(shared_node)
            else:
                shared_node = [v_id for v_id in assembly_network.get_element_end_point_ids(e_id)
                               if assembly_network.assembly_joints[v_id].is_grounded]
            assert(shared_node)

            e_ns = set(assembly_network.get_element_end_point_ids(e_id))
            e_ns.difference_update([shared_node[0]])
            way_points = interpolate_straight_line_pts(assembly_network.get_node_point(shared_node[0]),
                                                       assembly_network.get_node_point(e_ns.pop()),
                                                       WAYPOINT_DISC_LEN)

            e_body = assembly_network.get_element_body(e_id)
            cap_rung = CapRung()
            cap_rung.path_pts = way_points

            assert(seq_poses[i])
            cap_rung.ee_dirs = seq_poses[i]
            cap_rung.collision_fn = get_collision_fn(robot, get_movable_joints(robot), built_obstacles,
                                        attachments=[], self_collisions=SELF_COLLISIONS,
                                        disabled_collisions=disabled_collisions,
                                        custom_limits={})
            self.cap_rungs.append(cap_rung)
            built_obstacles.append(e_body)
            seq.update([e_id])

    def find_sparse_path(self, max_time=0.0):
        # if max_time < 5:
        #     # TODO: this timeout might be too long...
        #     max_time = DEFAULT_SPARSE_GRAPH_RUNG_TIMEOUT * len(self.cap_rungs)
        #     print('sparse graph sample time < 5s, set to default'.format(max_time))

        # find intial solution
        prev_vert = None

        for r_id, cap_rung in enumerate(self.cap_rungs):
            unit_time = time.time()
            cap_vert = CapVert(self.dof)

            while (time.time() - unit_time) < DEFAULT_UNIT_PROCESS_TIMEOUT:
                poses, cap_vert = generate_sample(cap_rung, cap_vert)

                f_cap_vert = check_cap_vert_feasibility(self.robot, poses, cap_rung, cap_vert)
                if f_cap_vert:
                    cap_vert = f_cap_vert
                    cap_vert.parent_vert = prev_vert
                    cap_vert.rung_id = r_id
                    cap_rung.cap_verts.append(cap_vert)
                    prev_vert = cap_vert
                    break

            if not cap_rung.cap_verts:
                print('cap_rung #{0} fails to find a feasible sol within timeout {1}'.format(r_id, DEFAULT_UNIT_PROCESS_TIMEOUT))
                return np.inf
            # else:
                # if DEBUG: print('cap_rung #{0}'.format(r_id))

        initial_cost = self.cap_rungs[-1].cap_verts[0].get_cost_to_root()
        print('initial sol found! cost: {}'.format(initial_cost))
        print('RRT* improv starts, comp time:{}'.format(max_time))

        rrt_st_time = time.time()
        while (time.time() - rrt_st_time) < max_time:
            rung_id_sample = random.choice(range(len(self.cap_rungs)))
            sample_rung = self.cap_rungs[rung_id_sample]
            new_vert = CapVert(self.dof)
            poses, new_vert = generate_sample(sample_rung, new_vert)

            new_vert = check_cap_vert_feasibility(self.robot, poses, sample_rung, new_vert)
            if new_vert:
                # find nearest node in tree
                c_min = np.inf
                nearest_vert = None
                if rung_id_sample > 0:
                    for near_vert in self.cap_rungs[rung_id_sample-1].cap_verts:
                        new_near_cost = near_vert.get_cost_to_root() + new_vert.distance_to(near_vert)
                        if c_min > new_near_cost:
                            nearest_vert = near_vert
                            c_min = new_near_cost

                # add new vert into the tree
                new_vert.rung_id = rung_id_sample
                new_vert.parent_vert = nearest_vert
                sample_rung.cap_verts.append(new_vert)

                # update vert on next rung (repair tree)
                if rung_id_sample < len(self.cap_rungs)-1:
                    new_vert_cost = new_vert.get_cost_to_root()
                    for next_vert in self.cap_rungs[rung_id_sample+1].cap_verts:
                        old_next_cost = next_vert.get_cost_to_root()
                        new_next_cost = new_vert_cost + next_vert.distance_to(new_vert)
                        if old_next_cost > new_next_cost:
                            next_vert.parent_vert = new_vert

        last_cap_vert = min(self.cap_rungs[-1].cap_verts, key = lambda x: x.get_cost_to_root())
        rrt_cost = last_cap_vert.get_cost_to_root()

        print('rrt* sol cost: {}'.format(rrt_cost))
        return rrt_cost

    def extract_solution(self):
        graphs = []
        graph_indices = []
        last_cap_vert = min(self.cap_rungs[-1].cap_verts, key = lambda x: x.get_cost_to_root())
        while last_cap_vert:
            cap_rung = self.cap_rungs[last_cap_vert.rung_id]
            poses = [multiply(Pose(point=pt), last_cap_vert.quat_pose) for pt in cap_rung.path_pts]
            unit_ladder_graph = generate_ladder_graph_from_poses(self.robot, self.dof, poses)
            assert(unit_ladder_graph)
            graphs = [unit_ladder_graph] + graphs
            graph_indices = [unit_ladder_graph.size] + graph_indices
            last_cap_vert = last_cap_vert.parent_vert

        unified_graph = LadderGraph(self.dof)
        for g in graphs:
            assert(g.dof == unified_graph.dof)
            unified_graph = append_ladder_graph(unified_graph, g)

        dag_search = DAGSearch(unified_graph)
        dag_search.run()
        tot_traj = dag_search.shortest_path()

        return tot_traj, graph_indices
