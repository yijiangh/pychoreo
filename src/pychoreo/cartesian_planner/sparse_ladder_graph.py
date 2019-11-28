import warnings
import time
import random

from pybullet_planning import INF, Pose
from pybullet_planning import multiply, wait_for_user

from pychoreo.utils import is_any_empty
from pychoreo.cartesian_planner.ladder_graph import LadderGraph, append_ladder_graph
from pychoreo.cartesian_planner.dag_search import DAGSearch
from pychoreo.cartesian_planner.ladder_graph_interface import generate_ladder_graph_from_poses
from pychoreo.cartesian_planner.postprocessing import divide_list_chunks
from pychoreo.process_model.trajectory import Trajectory

class CapVert(object):
    def __init__(self, dof, host_rung_id=None):
        self.dof = dof
        self._host_rung_id = host_rung_id
        self._st_jt_data = []
        self._end_jt_data = []
        self._to_parent_cost = INF
        self._parent_vert = None
        self._ee_poses = []
        # self.z_axis_angle = None
        # self.quat_pose = None

    @property
    def host_rung_id(self):
        return self._host_rung_id

    @host_rung_id.setter
    def host_rung_id(self, host_rung_id_):
        self._host_rung_id = host_rung_id_

    @property
    def st_jt_data(self):
        return self._st_jt_data

    @st_jt_data.setter
    def st_jt_data(self, st_jt_data_):
        self._st_jt_data = st_jt_data_

    @property
    def end_jt_data(self):
        return self._end_jt_data

    @end_jt_data.setter
    def end_jt_data(self, end_jt_data_):
        self._end_jt_data = end_jt_data_

    @property
    def ee_poses(self):
        return self._ee_poses

    @ee_poses.setter
    def ee_poses(self, ee_poses_):
        self._ee_poses = ee_poses_

    def distance_to(self, v):
        """compute distance to CapVert v.
        The distance is defined by ...

        Parameters
        ----------
        v : CapVert

        Returns
        -------
        float
            distance between the current CapVert and v
        """
        if not v:
            return 0
        assert(isinstance(v, CapVert))
        assert(self.dof == v.dof)
        cost = INF
        dof = self.dof
        # TODO: maybe we can get rid of the dof and concatenation here
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
        return self._to_parent_cost

    @parent_cost.setter
    def parent_cost(self, c):
        self._to_parent_cost = c

    @property
    def parent_vert(self):
        return self._parent_vert

    @parent_vert.setter
    def parent_vert(self, v):
        """Set the parent vert of current CapVert (with associated edge cost)

        Parameters
        ----------
        v : CapVert
            Parent capsule vertex.
        """
        assert(isinstance(v, CapVert) or v == None)
        self._parent_vert = v
        self.parent_cost = self.distance_to(v)

    def get_cost_to_root(self):
        prev_v = self.parent_vert
        cost = self.parent_cost
        while prev_v:
            if prev_v.parent_vert:
                cost += prev_v.parent_cost
            prev_v = prev_v.parent_vert
        return cost

    def __repr__(self):
        return 'CapVert r_id:{}|stJ#{}|endJ#{}|parent_cost:{}|parent rung:{}'.format(
            self.host_rung_id, len(self.st_jt_data), len(self.end_jt_data), self.parent_cost, self.parent_vert.host_rung_id if self.parent_vert else -1)

class CapRung(object):
    """The CapRung (capsulated ladder rung) is an abstract containter for
    CapVerts that corresponds to the same Cartesian process to live in.
    A CapRung have all the information needed to sample end effector poses and
    check feasibility to generate new CapVert or to certify the feasibility of
    a given CapRung.

    """
    def __init__(self, cart_proc=None, rung_id=None):
        self._rung_id = rung_id
        self._cap_verts = []
        self._cart_proc = cart_proc
        # self.path_pts = []
        # # all the candidate orientations for each kinematics segment
        # self.ee_dirs = []
        # self.obstacles = [] # TODO: store index, point to a shared list of obstacles
        # self.collision_fn = None

    @property
    def dof(self):
        return self.cartesian_process.dof

    @property
    def rung_id(self):
        return self._rung_id

    @property
    def cap_verts(self):
        return self._cap_verts

    @property
    def cartesian_process(self):
        return self._cart_proc

    @cartesian_process.setter
    def cartesian_process(self, cartesian_proc_):
        self._cart_proc = cartesian_proc_

    def sample_cap_vert(self, check_collision=True):
        try:
            ee_poses = self.cartesian_process.sample_ee_poses()
        except StopIteration:
            warnings.warn('ee pose gen fn exhausted, we should not plug a finite generator in the SparseLadderGraph. Iterator reset.')
            self.cartesian_process.reset_ee_pose_gen_fn()
            ee_poses = self.cartesian_process.sample_ee_poses()
        ik_sols = self.cartesian_process.get_ik_sols(ee_poses, check_collision=check_collision)
        if is_any_empty(ik_sols):
            return None
        else:
            cap_vert = CapVert(self.dof, host_rung_id=self.rung_id)
            cap_vert.st_jt_data = [jv for jts in ik_sols[0][0] for jv in jts]
            cap_vert.end_jt_data = [jv for jts in ik_sols[-1][-1] for jv in jts]
            cap_vert.ee_poses = ee_poses
            return cap_vert

class SparseLadderGraph(object):
    def __init__(self, cart_proc_list):
        assert len(cart_proc_list) > 0 and isinstance(cart_proc_list, list)
        self._cap_rungs = [CapRung(cart_proc=cart_proc, rung_id=cp_id) for cp_id, cart_proc in enumerate(cart_proc_list)]
        # self._cap_rungs = []
        # for cp_id, cart_proc in enumerate(cart_proc_list):
        #     self._cap_rungs.append(CapRung(cart_proc=cart_proc, rung_id=cp_id))
        self._cart_proc_list = cart_proc_list

    @classmethod
    def from_cartesian_process_list(cls, cart_proc_list):
        return cls(cart_proc_list)

    @property
    def cap_rungs(self):
        return self._cap_rungs

    @property
    def cart_proc_list(self):
        return self._cart_proc_list

    def find_sparse_path(self, check_collision=True, vert_timeout=2.0, sparse_sample_timeout=5.0, verbose=False):
        if verbose:
            print('sparse graph vert sample timeout: {}, sparse graph sampling timeout : {}'.format(
                vert_timeout, sparse_sample_timeout))

        # find an intial solution
        init_sol_st_time = time.time()
        prev_vert = None
        for r_id, cap_rung in enumerate(self.cap_rungs):
            unit_st_time = time.time()
            while (time.time() - unit_st_time) < vert_timeout:
                cap_vert = cap_rung.sample_cap_vert(check_collision=check_collision)
                if cap_vert:
                    # if one feasible instance of cap_vert in this rung has been found, break the loop
                    cap_vert.parent_vert = prev_vert
                    cap_vert.host_rung_id = r_id
                    cap_rung.cap_verts.append(cap_vert)
                    prev_vert = cap_vert
                    break

            if not cap_rung.cap_verts:
                print('cap_rung #{0} fails to find a feasible sol within timeout {1}'.format(r_id, vert_timeout))
                return INF
            else:
                if verbose: print('cap_rung #{0} has found an initial feasible cap_vert.'.format(r_id))

        initial_cost = self.cap_rungs[-1].cap_verts[0].get_cost_to_root()
        if verbose:
            print('initial sol found in {} sec! cost: {}'.format(time.time()-init_sol_st_time, initial_cost))
            print('RRT* improv starts, comp time:{}'.format(sparse_sample_timeout))

        rrt_st_time = time.time()
        while (time.time() - rrt_st_time) < sparse_sample_timeout:
            rung_id_sample = random.choice(range(len(self.cap_rungs)))
            sampled_rung = self.cap_rungs[rung_id_sample]
            new_vert = sampled_rung.sample_cap_vert(check_collision=check_collision)
            if new_vert:
                # find nearest node in tree
                c_min = INF
                nearest_vert = None
                if rung_id_sample > 0:
                    for near_vert in self.cap_rungs[rung_id_sample-1].cap_verts:
                        new_near_cost = near_vert.get_cost_to_root() + new_vert.distance_to(near_vert)
                        if c_min > new_near_cost:
                            nearest_vert = near_vert
                            c_min = new_near_cost

                # add new vert into the tree
                new_vert.host_rung_id = rung_id_sample
                new_vert.parent_vert = nearest_vert
                sampled_rung.cap_verts.append(new_vert)

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

        if verbose: print('Sparse ladder graph done: rrt* sol cost: {}'.format(rrt_cost))
        return rrt_cost

    def extract_solution(self, check_collision=True, verbose=False, warning_pause=False):
        """extract ladder graph solution out of a solved sparse path

        Returns
        -------
        a list of CartesianProcess
            with trajectory filled in.
        """
        if verbose: st_time = time.time()
        graph_dict = {}
        last_cap_vert = min(self.cap_rungs[-1].cap_verts, key = lambda x: x.get_cost_to_root())
        while last_cap_vert:
            cap_rung = self.cap_rungs[last_cap_vert.host_rung_id]
            # TODO: recover full list of path points from the capsulated vertex
            # poses = [multiply(Pose(point=pt), last_cap_vert.quat_pose) for pt in cap_rung.path_pts]
            unit_ladder_graph = generate_ladder_graph_from_poses(
                cap_rung.cartesian_process, last_cap_vert.ee_poses, check_collision=check_collision)
            if unit_ladder_graph and unit_ladder_graph.size > 0:
                graph_dict[cap_rung.rung_id] = unit_ladder_graph
                if verbose: print('#{}-{} ladder graph formed.'.format(cap_rung.rung_id, cap_rung.cartesian_process))
            else:
                warnings.warn('Warning: cart proce #{}-{} does not have any valid joint sols to form rungs!'.format(
                    cap_rung.rung_id, cap_rung.cartesian_process))
                if warning_pause : wait_for_user()
                assert(unit_ladder_graph)
            last_cap_vert = last_cap_vert.parent_vert

        # * horizontally concatenate the graphs
        unified_graph = LadderGraph(unit_ladder_graph.dof)
        for cp_id in sorted(graph_dict):
            g = graph_dict[cp_id]
            unified_graph = append_ladder_graph(unified_graph, g)
        if verbose: print('ladder graph formed in {} secs (rung size #{}), proceed to DAG search.'.format(time.time()-st_time, unified_graph.get_rungs_size()))

        # * DAG solve for the concatenated graph
        st_time = time.time()
        dag_search = DAGSearch(unified_graph)
        min_cost = dag_search.run()
        tot_traj = dag_search.shortest_path()
        if verbose: print('DAG search done in {} secs, cost {}.'.format(time.time()-st_time, min_cost))

        # * Divide the contatenated trajectory back to processes
        proc_trajs = divide_list_chunks(tot_traj, [graph_dict[cp_id].size for cp_id in sorted(graph_dict)])
        proc_trajs = {cp_id : traj for cp_id, traj in zip(sorted(graph_dict), proc_trajs)}
        for cp_id, proc_traj in proc_trajs.items():
            # divide into subprocesses
            subp_trajs = divide_list_chunks(proc_traj, [sp.path_point_size for sp in self.cart_proc_list[cp_id].sub_process_list])
            for sp, subp_traj in zip(self.cart_proc_list[cp_id].sub_process_list, subp_trajs):
                if sp.trajectory is None:
                    sp.trajectory = Trajectory(self.cart_proc_list[cp_id].robot, self.cart_proc_list[cp_id].ik_joints, subp_traj)
                else:
                    sp.trajectory.traj_path = subp_traj
        return self.cart_proc_list
