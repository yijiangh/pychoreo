import numpy as np
import time
import random
from copy import deepcopy
from .choreo_utils import WAYPOINT_DISC_LEN, interpolate_straight_line_pts, get_collision_fn, generate_way_point_poses, \
    make_print_pose, sample_ee_yaw, interpolate_cartesian_poses

# from conrob_pybullet.utils.ikfast.kuka_kr6_r900.ik import sample_tool_ik
from conrob_pybullet.utils.ikfast.abb_irb6600_track.ik import sample_tool_ik, get_track_arm_joints

from assembly_datastructure import AssemblyNetwork, Brick
from conrob_pybullet.ss_pybullet.pybullet_tools.utils import Pose, \
    get_movable_joints, multiply, Attachment, set_pose, invert, draw_pose, wait_for_interrupt, set_joint_positions, \
    wait_for_user
from choreo.extrusion_utils import get_disabled_collisions

DEBUG=True
DEFAULT_UNIT_PROCESS_TIMEOUT = 10
DEFAULT_SPARSE_GRAPH_RUNG_TIMEOUT = 4
SELF_COLLISIONS=False
# utils

def divide_list_chunks(list, size_list):
    assert(sum(size_list) >= len(list))
    if sum(size_list) < len(list):
        size_list.append(len(list) - sum(size_list))
    for j in range(len(size_list)):
        cur_id = sum(size_list[0:j])
        yield list[cur_id:cur_id+size_list[j]]

#########################################

class LadderGraphEdge(object):
    def __init__(self, idx=None, cost=-np.inf):
        self.idx = idx # the id of the destination vert
        self.cost = cost

    def __repr__(self):
        return 'E idx{0}, cost{1}'.format(self.idx, self.cost)


class LadderGraphRung(object):
    def __init__(self, id=None, data=[], edges=[], collision_fn=None):
        self.id = id
        self.data = data # joint_data: joint values are stored in one contiguous list
        self.edges = edges

        #TODO: collision bodies or collision fn
        self.collision_fn = collision_fn

    def __repr__(self):
        return 'id {0}, data {1}, edge num {2}'.format(self.id, len(self.data), len(self.edges))

# TODO: we ignore the timing constraint here

class LadderGraph(object):
    def __init__(self, dof):
        assert(dof != 0)
        self.dof = dof
        self.rungs = []

    def get_dof(self):
        return self.dof

    def get_rung(self, rung_id):
        assert(rung_id < len(self.rungs))
        return self.rungs[rung_id]

    def get_edges(self, rung_id):
        return self.get_rung(rung_id).edges

    def get_edge_sizes(self):
        return [len(r.edges) for r in self.rungs]

    def get_data(self, rung_id):
        return self.get_rung(rung_id).data

    def get_rungs_size(self):
        return len(self.rungs)

    @property
    def size(self):
        return self.get_rungs_size()

    def get_rung_vert_size(self, rung_id):
        """count the number of vertices in a rung"""
        return len(self.get_rung(rung_id).data) / self.dof

    def get_vert_size(self):
        """count the number of vertices in the whole graph"""
        return sum([self.get_rung_vert_size(r_id) for r_id in range(self.get_rungs_size())])

    def get_vert_sizes(self):
        return [self.get_rung_vert_size(r_id) for r_id in range(self.get_rungs_size())]

    def get_vert_data(self, rung_id, vert_id):
        return self.get_rung(rung_id).data[self.dof * vert_id : self.dof * (vert_id+1)]

    def resize(self, rung_number):
        if self.size == 0:
            self.rungs = [LadderGraphRung(id=None, data=[], edges=[]) for i in range(rung_number)]
            return
        if self.size > 0 and self.size < rung_number:
            # fill in the missing ones with empty rungs
            self.rungs.extend([LadderGraphRung(id=None, data=[], edges=[]) for i in range(rung_number - self.size)])
            return
        elif self.size > rung_number:
            self.rungs = [r for i, r in enumerate(self.rungs) if i < rung_number]
            return

    def clear(self):
        self.rungs = []

    # assign fns
    def assign_rung(self, r_id, sol_lists):
        rung = self.get_rung(r_id)
        rung.id = r_id
        rung.data = [jt for jt_l in sol_lists for jt in jt_l]
        assert(len(rung.data) % self.dof == 0)

    def assign_edges(self, r_id, edges):
        # edges_ref = self.get_edges(r_id)
        self.get_rung(r_id).edges = edges

    def __repr__(self):
        return 'g tot_r_size:{0}, v_sizes:{1}, e_sizes:{2}'.format(self.size, self.get_vert_sizes(), self.get_edge_sizes())

    # insert_rung, clear_rung_edges (maybe not needed at all)

class EdgeBuilder(object):
    """edge builder for ladder graph, construct edges for fully connected biparte graph"""
    def __init__(self, n_start, n_end, dof, upper_tm=None, joint_vel_limits=None):
        self.result_edges_ = [[] for i in range(n_start)]
        self.edge_scratch_ = [LadderGraphEdge(idx=None, cost=None) for i in range(n_end)] # preallocated space to work on
        self.dof_ = dof
        self.delta_buffer_ = [0] * dof
        self.count_ = 0
        self.has_edges_ = False

    def consider(self, st_jt, end_jt, index):
        """index: to_id"""
        # TODO check delta joint val exceeds the joint_vel_limits
        for i in range(self.dof_):
            self.delta_buffer_[i] = abs(st_jt[i] - end_jt[i])
        cost = sum(self.delta_buffer_)
        assert(self.count_ < len(self.edge_scratch_))
        self.edge_scratch_[self.count_].cost = cost
        self.edge_scratch_[self.count_].idx = index
        self.count_ += 1

    def next(self, i):
        #TODO: want to do std::move here to transfer memory...
        self.result_edges_[i] = deepcopy(self.edge_scratch_)
        self.has_edges_ = self.has_edges_ or self.count_ > 0
        self.count_ = 0

    @property
    def result(self):
        return self.result_edges_

    @property
    def has_edges(self):
        return self.has_edges_


def generate_ladder_graph_from_poses(robot, dof, pose_list, collision_fn=lambda x: False, dt=-1):
    # TODO: lazy collision check
    # TODO: dt, timing constraint
    graph = LadderGraph(dof)
    graph.resize(len(pose_list))

    # solve ik for each pose, build all rungs (w/o edges)
    for i, pose in enumerate(pose_list):
        jt_list = sample_tool_ik(robot, pose, get_all=True)
        jt_list = [jts for jts in jt_list if jts and not collision_fn(jts)]
        if not jt_list or all(not jts for jts in jt_list):
           return None
        graph.assign_rung(i, jt_list)

    # build edges
    for i in range(graph.get_rungs_size()-1):
        st_id = i
        end_id = i + 1
        jt1_list = graph.get_data(st_id)
        jt2_list = graph.get_data(end_id)
        st_size = graph.get_rung_vert_size(st_id)
        end_size = graph.get_rung_vert_size(end_id)
        edge_builder = EdgeBuilder(st_size, end_size, dof)

        for k in range(st_size):
            st_id = k * dof
            for j in range(end_size):
                end_id = j * dof
                edge_builder.consider(jt1_list[st_id : st_id+dof], jt2_list[end_id : end_id+dof], j)
            edge_builder.next(k)

        edges = edge_builder.result
        if not edge_builder.has_edges and DEBUG:
            print('no edges!')

        graph.assign_edges(i, edges)
    return graph


def concatenate_graph_vertically(graph_above, graph_below):
    assert isinstance(graph_above, LadderGraph)
    assert isinstance(graph_below, LadderGraph)
    assert graph_above.size() == graph_below.size() # same number of rungs
    num_rungs = graph_above.size()
    for i in range(num_rungs):
        rung_above = graph_above.get_rung(i)
        above_jts = graph_above.get_rung(i).data
        below_jts = graph_below.get_rung(i).data
        above_jts.extend(below_jts)
        if i != num_rungs - 1:
            # shifting target vert id in below_edges
            next_above_rung_size = graph_below.get_rung_vert_size(i + 1)
            below_edges = graph_below.get_edges(i)
            for e in below_edges:
                e_copy = deepcopy(e)
                e_copy.idx += next_above_rung_size
                rung_above.edges.append(e_copy)
    return graph_above


class SolutionRung(object):
    def __init__(self):
        self.distance = []
        self.predecessor = []

    def extract_min(self):
        min_dist = min(self.distance)
        min_id = self.distance.index(min_dist)
        return min_dist, min_id

    def __len__(self):
        assert(len(self.distance) == len(self.predecessor))
        return len(self.distance)
    #
    # def __repr__(self):
    #     return 'min dist: {0}, min pred id: {1}'.format(self.extract_min())

class DAGSearch(object):
    def __init__(self, graph):
        assert(isinstance(graph, LadderGraph))
        self.graph = graph
        self.solution = [SolutionRung() for i in range(graph.get_rungs_size())]

        # allocate everything we need
        for i in range(graph.get_rungs_size()):
            n_verts = graph.get_rung_vert_size(i)
            assert(n_verts > 0)
            self.solution[i].distance = [0] * n_verts
            self.solution[i].predecessor = [0] * n_verts

    def distance(self, r_id, v_id):
        return self.solution[r_id].distance[v_id]

    def predecessor(self, r_id, v_id):
        return self.solution[r_id].predecessor[v_id]

    def run(self):
        """forward cost propagation"""
        # first rung init to 0
        self.solution[0].distance = [0] * len(self.solution[0])

        # other rungs init to inf
        for j in range(1, len(self.solution)):
            self.solution[j].distance = [np.inf] * len(self.solution[j])

        for r_id in range(0, len(self.solution)-1):
            n_verts = self.graph.get_rung_vert_size(r_id)
            next_r_id = r_id + 1
            # for each vert in the out edge list
            for v_id in range(n_verts):
                u_cost = self.distance(r_id, v_id)
                edges = self.graph.get_edges(r_id)[v_id]
                for edge in edges:
                    dv = u_cost + edge.cost
                    if dv < self.distance(next_r_id, edge.idx):
                        self.solution[next_r_id].distance[edge.idx] = dv
                        self.solution[next_r_id].predecessor[edge.idx] = v_id

        return min(self.solution[-1].distance)

    def shortest_path(self):
        _, min_val_id = self.solution[-1].extract_min()
        min_idx = self.solution[-1].predecessor[min_val_id]
        path_idx = [0] * len(self.solution)

        current_v_id = min_idx
        for i in range(len(path_idx)):
            count = len(path_idx) - 1 - i
            path_idx[count] = current_v_id
            current_v_id = self.predecessor(count, current_v_id)

        sol = []
        for r_id, v_id in enumerate(path_idx):
            data = self.graph.get_vert_data(r_id, v_id)
            sol.append(data)

        return sol

def append_ladder_graph(current_graph, next_graph):
    assert(isinstance(current_graph, LadderGraph) and isinstance(next_graph, LadderGraph))
    assert(current_graph.dof == next_graph.dof)

    cur_size = current_graph.size
    new_tot_size = cur_size + next_graph.size
    dof = current_graph.dof

    # just add two sets of rungs together to have a longer ladder graph
    current_graph.resize(new_tot_size)
    for i in range(next_graph.size):
        current_graph.rungs[cur_size + i] = next_graph.rungs[i]

    # connect graphs at the boundary
    a_rung = current_graph.get_rung(cur_size - 1)
    b_rung = current_graph.get_rung(cur_size)
    n_st_vert = len(a_rung.data) / dof
    n_end_vert = len(b_rung.data) / dof

    edge_builder = EdgeBuilder(n_st_vert, n_end_vert, dof)
    for k in range(n_st_vert):
        st_id = k * dof
        for j in range(n_end_vert):
            end_id = j * dof
            edge_builder.consider(a_rung.data[st_id : st_id+dof], b_rung.data[end_id : end_id+dof], j)
        edge_builder.next(k)

    edges_list = edge_builder.result
    # assert(edge_builder.has_edges)
    current_graph.assign_edges(cur_size - 1, edges_list)
    return current_graph

#########################
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
        n_prev_end = len(v.end_jt_data) / dof
        n_this_st = len(self.st_jt_data) / dof

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
        disabled_collisions = get_disabled_collisions(robot)
        built_obstacles = static_obstacles

        seq = set()
        for i in element_seq.keys():
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

def direct_ladder_graph_solve(robot, assembly_network, element_seq, seq_poses, obstacles, assembly_type='picknplace'):
    dof = len(get_movable_joints(robot))
    graph_list = []
    built_obstacles = obstacles
    disabled_collisions = get_disabled_collisions(robot)

    for i in element_seq.keys():
           e_id = element_seq[i]
           p1, p2 = assembly_network.get_end_points(e_id)

           collision_fn = get_collision_fn(robot, get_movable_joints(robot), built_obstacles,
                                           attachments=[], self_collisions=SELF_COLLISIONS,
                                           disabled_collisions=disabled_collisions,
                                           custom_limits={} )
           st_time = time.time()
           while True:
               ee_dir = random.choice(seq_poses[i])
               rot_angle = random.uniform(-np.pi, np.pi)
               way_poses = generate_way_point_poses(p1, p2, ee_dir.phi, ee_dir.theta, rot_angle, WAYPOINT_DISC_LEN)
               graph = generate_ladder_graph_from_poses(robot, dof, way_poses, collision_fn)
               if graph is not None:
                   # print(graph)
                   graph_list.append(graph)
                   break
               if time.time() - st_time > 5:
                   break
           built_obstacles.append(assembly_network.get_element_body(e_id))

    unified_graph = LadderGraph(dof)
    for g in graph_list:
        unified_graph = append_ladder_graph(unified_graph, g)

    dag_search = DAGSearch(unified_graph)
    dag_search.run()
    graph_sizes = [g.size for g in graph_list]
    tot_traj = dag_search.shortest_path()
    return tot_traj, graph_sizes


def generate_ladder_graph_for_picknplace_single_brick(robot, dof, brick, disc_len, tool_link, obstacles):
    # TODO: lazy collision check
    # TODO: dt, timing constraint

    # assert(isinstance(brick, Brick))
    vertical_graph = LadderGraph(dof)
    disabled_collisions = get_disabled_collisions(robot)
    movable_joints = get_track_arm_joints(robot)

    # generate path pts
    grasps = brick.grasps
    for grasp in grasps:
        print(grasp)
        def make_assembly_poses(obj_pose, grasp_poses):
            return [multiply(obj_pose, g_pose) for g_pose in grasp_poses]

        world_from_pick_poses = make_assembly_poses(brick.initial_pose, [grasp.approach, grasp.attach, grasp.retreat])
        world_from_place_poses = make_assembly_poses(brick.goal_pose, [grasp.approach, grasp.attach, grasp.retreat])

        approach2attach_pick = interpolate_cartesian_poses(world_from_pick_poses[0], world_from_pick_poses[1], disc_len)
        attach2retreat_pick = interpolate_cartesian_poses(world_from_pick_poses[1], world_from_pick_poses[2], disc_len)
        approach2attach_place = interpolate_cartesian_poses(world_from_place_poses[0], world_from_place_poses[1], disc_len)
        attach2retreat_place = interpolate_cartesian_poses(world_from_place_poses[1], world_from_place_poses[2], disc_len)

        picknplace_pose_lists = [approach2attach_pick] + [attach2retreat_pick] + \
                        [approach2attach_place] + [attach2retreat_place]
        accum_sub_id = 0
        process_map = {}
        picknplace_poses = []
        for sub_id, sub_path in enumerate(picknplace_pose_lists):
            for pose_id, pose in enumerate(sub_path):
                process_map[accum_sub_id + pose_id] = sub_id
                picknplace_poses.append(pose)
            accum_sub_id += len(sub_path)

        for p_tmp in picknplace_poses:
            # print(p_tmp)
            draw_pose(p_tmp, length=0.04)

        collision_fns = []
        def dummy_collision_fn():
            return False

        attachment = Attachment(robot, tool_link, invert(grasp.attach), brick.body)
        collision_fns = [dummy_collision_fn, dummy_collision_fn, dummy_collision_fn, dummy_collision_fn]
        # collision_fns.append(get_collision_fn(robot, get_movable_joints(robot), obstacles + [brick.body],
        #                                       attachments=[], self_collisions=SELF_COLLISIONS,
        #                                       disabled_collisions=disabled_collisions,
        #                                       custom_limits={}))
        # collision_fns.append(get_collision_fn(robot, get_movable_joints(robot), obstacles,
        #                                       attachments=[attachment], self_collisions=SELF_COLLISIONS,
        #                                       disabled_collisions=disabled_collisions,
        #                                       custom_limits={}))
        # collision_fns.append(get_collision_fn(robot, get_movable_joints(robot), obstacles,
        #                                       attachments=[attachment], self_collisions=SELF_COLLISIONS,
        #                                       disabled_collisions=disabled_collisions,
        #                                       custom_limits={}))
        # collision_fns.append(get_collision_fn(robot, get_movable_joints(robot), obstacles + [brick.body],
        #                                       attachments=[], self_collisions=SELF_COLLISIONS,
        #                                       disabled_collisions=disabled_collisions,
        #                                       custom_limits={}))

        graph = LadderGraph(dof)
        graph.resize(len(picknplace_poses))
        is_empty = False
        # solve ik for each pose, build all rungs (w/o edges)
        for i, pose in enumerate(picknplace_poses):
            # TODO: special sampler for 6+ extra dofs
            jt_list = sample_tool_ik(robot, pose, get_all=True, max_attempts=1000)
            # jt_list = [jts for jts in jt_list if jts and not collision_fns[process_map[i]](jts)]
            jt_list = [jts for jts in jt_list]
            # print(jt_list)
            if not jt_list or all(not jts for jts in jt_list):
                # print('no joint solution found at brick #{0} path pt #{1} grasp id #{2}'.format(brick.index, i, grasp.num))
                is_empty = True
                # break
            else:
                draw_pose(pose, length=0.04)
                print(len(jt_list))
                set_joint_positions(robot, movable_joints, jt_list[0])
                print('rung #{0} at brick #{1} grasp id #{2}'.format(i, brick.index, grasp.num))
                wait_for_user()

                graph.assign_rung(i, jt_list)

        # if is_empty:
        #     continue

        print('Found!!! at brick #{0} grasp id #{1}'.format(brick.index, grasp.num))
        # wait_for_user()
        # build edges
        # for i in range(graph.get_rungs_size()-1):
        #     st_id = i
        #     end_id = i + 1
        #     jt1_list = graph.get_data(st_id)
        #     jt2_list = graph.get_data(end_id)
        #     st_size = graph.get_rung_vert_size(st_id)
        #     end_size = graph.get_rung_vert_size(end_id)
        #     edge_builder = EdgeBuilder(st_size, end_size, dof)
        #
        #     for k in range(st_size):
        #         st_id = k * dof
        #         for j in range(end_size):
        #             end_id = j * dof
        #             edge_builder.consider(jt1_list[st_id : st_id+dof], jt2_list[end_id : end_id+dof], j)
        #         edge_builder.next(k)
        #
        #     edges = edge_builder.result
        #     if not edge_builder.has_edges and DEBUG:
        #         print('no edges!')
        #
        #     graph.assign_edges(i, edges)

        print(graph)
        # concatenate_graph_vertically(vertical_graph, graph)
    return vertical_graph


def direct_ladder_graph_solve_picknplace(robot, brick_from_index, element_seq, obstacle_from_name, tool_link):
    dof = len(get_movable_joints(robot))
    graph_list = []
    built_obstacles = list(obstacle_from_name.values())

    for seq_id, e_id in element_seq.items():
        brick = brick_from_index[e_id]

        graph = generate_ladder_graph_for_picknplace_single_brick(robot, dof, brick, WAYPOINT_DISC_LEN, tool_link, built_obstacles)
        if graph is not None:
            # print(graph)
            graph_list.append(graph)
        else:
            assert('graph empty at brick #{0} at seq #{1}'.format(e_id, seq_id))

        set_pose(brick.body, brick.goal_pose)
        built_obstacles.append(brick.body)

    print(graph_list)

    unified_graph = LadderGraph(dof)
    # for g in graph_list:
    #     unified_graph = append_ladder_graph(unified_graph, g)

    # dag_search = DAGSearch(unified_graph)
    # dag_search.run()
    # graph_sizes = [g.size for g in graph_list]
    # tot_traj = dag_search.shortest_path()

    graph_sizes = []
    tot_traj = []
    return tot_traj, graph_sizes

def generate_hybrid_motion_plans():
    # ladder graph to solve semi-constrained motion during extrusion

    # retraction planning (omitted for now)

    # transition planning

    pass
