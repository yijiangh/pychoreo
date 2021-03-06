import warnings
import time
from pybullet_planning import WorldSaver
from pybullet_planning import joints_from_names, set_joint_positions, wait_for_user

from pychoreo.utils import is_any_empty
from pychoreo.cartesian_planner.ladder_graph import LadderGraph, EdgeBuilder
from pychoreo.cartesian_planner.ladder_graph import append_ladder_graph, concatenate_graph_vertically
from pychoreo.cartesian_planner.dag_search import DAGSearch
from pychoreo.cartesian_planner.postprocessing import divide_list_chunks
from pychoreo.process_model.trajectory import Trajectory

def solve_ladder_graph_from_cartesian_process_list(cart_proc_list, start_conf=None, check_collision=True, verbose=False, viz_inspect=False, warning_pause=True):
    world_saver = WorldSaver()
    st_time = time.time()
    if verbose: print('Start building ladder graph.')
    # * build ladder graph for each cart_proc in the list
    graph_dict = {}
    for cp_id, cart_proc in enumerate(cart_proc_list):
        if verbose: print('#{}: {}'.format(cp_id, cart_proc))
        vertical_graph, vertical_subgraph_cnt = generate_ladder_graph_from_cartesian_process(cart_proc, check_collision=check_collision, viz_inspect=viz_inspect)

        if vertical_graph.size > 0:
            graph_dict[cp_id] = vertical_graph
            if verbose: print('#{}-{} #{} pose families formed.'.format(cp_id, cart_proc, vertical_subgraph_cnt))
        else:
            warnings.warn('Warning: cart proce #{}-{} does not have any valid joint sols to form rungs!'.format(cp_id, cart_proc))
            if warning_pause : wait_for_user()
        # end loop candidates poses for process
    # end loop all processes
    world_saver.restore()

    # * horizontally concatenate the graphs
    unified_graph = LadderGraph(vertical_graph.dof)
    if start_conf:
        assert len(start_conf) == vertical_graph.dof
        st_graph = LadderGraph(vertical_graph.dof)
        st_graph.resize(1)
        st_graph.assign_rung(0, [start_conf])
        unified_graph = append_ladder_graph(unified_graph, st_graph)
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
    if start_conf:
        del tot_traj[0]

    # * Divide the contatenated trajectory back to processes
    proc_trajs = divide_list_chunks(tot_traj, [g.size for g in graph_dict.values()])
    proc_trajs = {cp_id : traj for cp_id, traj in zip(graph_dict.keys(), proc_trajs)}
    print({proc_id : len(val) for proc_id, val in proc_trajs.items()})
    for cp_id, proc_traj in proc_trajs.items():
        # divide into subprocesses
        subp_trajs = divide_list_chunks(proc_traj, [sp.path_point_size for sp in cart_proc_list[cp_id].sub_process_list])
        for sp, subp_traj in zip(cart_proc_list[cp_id].sub_process_list, subp_trajs):
            if sp.trajectory is None:
                sp.trajectory = Trajectory(cart_proc_list[cp_id].robot, cart_proc_list[cp_id].ik_joints, subp_traj)
            else:
                sp.trajectory.traj_path = subp_traj
    return cart_proc_list

def generate_ladder_graph_from_cartesian_process(cart_proc, check_collision=True, viz_inspect=False):
    vertical_graph = LadderGraph(cart_proc.dof)
    vertical_subgraph_cnt = 0
    for proc_ee_poses in cart_proc.exhaust_iter():
        graph = generate_ladder_graph_from_poses(cart_proc, proc_ee_poses, check_collision=check_collision, viz_inspect=viz_inspect)
        # vertically concatenate graphs, no extra edges added
        if graph and graph.size > 0:
            if vertical_graph.size == 0:
                # no graph from above
                vertical_graph = graph
            else :
                concatenate_graph_vertically(vertical_graph, graph)
            vertical_subgraph_cnt += 1
    return vertical_graph, vertical_subgraph_cnt

def generate_ladder_graph_from_poses(cart_proc, proc_ee_poses, check_collision=True, viz_inspect=False):
    proc_ik_sols = cart_proc.get_ik_sols(proc_ee_poses, check_collision=check_collision)
    # flatten ik sols of subprocesses, subprocess semantics can be recovered later based on numbers
    ik_sols = [jts for sp_ik_sols in proc_ik_sols for jts in sp_ik_sols]

    if is_any_empty(ik_sols):
        # if verbose:
        #     print('no joint solution found at {}'.format(cart_proc))
        return None
    else:
        dof = cart_proc.dof
        robot = cart_proc.robot
        ik_joint_names = cart_proc.ik_joint_names
        ik_joints = joints_from_names(robot, ik_joint_names)

        preference_cost = cart_proc.preference_cost_eval_fn(proc_ee_poses)

        graph = LadderGraph(dof)
        graph.resize(len(ik_sols))
        # visualize jt sol
        if viz_inspect:
            for ik_sol in ik_sols:
                for ik_jts in ik_sol:
                    set_joint_positions(robot, ik_joints, ik_jts)
                    wait_for_user()

        # assign rung data
        for pt_id, ik_jts_pt in enumerate(ik_sols):
            graph.assign_rung(pt_id, ik_jts_pt)

        # build edges within current pose family
        for i in range(graph.get_rungs_size()-1):
            st_id = i
            end_id = i + 1
            jt1_list = graph.get_data(st_id)
            jt2_list = graph.get_data(end_id)
            st_size = graph.get_rung_vert_size(st_id)
            end_size = graph.get_rung_vert_size(end_id)
            if st_size == 0 or end_size == 0:
                print(ik_sols)

            assert st_size > 0, 'Ladder graph not valid: rung {}/{} is a zero size rung'.format(st_id, graph.get_rungs_size())
            assert end_size > 0, 'Ladder graph not valid: rung {}/{} is a zero size rung'.format(end_id, graph.get_rungs_size())

            edge_builder = EdgeBuilder(st_size, end_size, dof, preference_cost=preference_cost)
            # edge_builder = EdgeBuilder(st_size, end_size, dof)
            for k in range(st_size):
                st_id = k * dof
                for j in range(end_size):
                    end_id = j * dof
                    edge_builder.consider(jt1_list[st_id : st_id+dof], jt2_list[end_id : end_id+dof], j)
                edge_builder.next(k)
            edges = edge_builder.result
            # if not edge_builder.has_edges and verbose:
            #     # TODO: more report information here
            #     print('no edges!')
            graph.assign_edges(i, edges)
        return graph
