from pybullet_planning import WorldSaver

from pychoreo.cartesian_planner.ladder_graph import LadderGraph, EdgeBuilder
from pychoreo.cartesian_planner.ladder_graph import append_ladder_graph, concatenate_graph_vertically
from pychoreo.cartesian_planner.dag_search import DAGSearch

def solve_ladder_graph_from_cartesian_processes(cart_proc_list, verbose=False, viz_inspect=False):
    # input can be list of lists
    # these will be concatenated horizontally under same parametrization
    # TODO: multiple cartesian processes share the same EE pose gen

    from pybullet_planning import joints_from_names, set_joint_positions, wait_for_user
    robot = cart_proc_list[0].robot
    ik_joint_names = cart_proc_list[0].ik_joint_names
    ik_joints = joints_from_names(robot, ik_joint_names)
    dof = cart_proc_list[0].dof

    world_saver = WorldSaver()
    # * build ladder graph for each cart_proc in the list
    graph_list = []
    for cp_id, cart_proc in enumerate(cart_proc_list):
        dof = cart_proc.dof
        vertical_graph = LadderGraph(dof)
        if verbose: print('#{}: {}'.format(cp_id, cart_proc))
        vertical_subgraph_cnt = 0
        while True:
            try:
                ee_poses = cart_proc.sample_ee_poses()
                ik_sols = cart_proc.get_ik_sols(ee_poses, check_collision=True)
                graph = LadderGraph(dof)

                if not ik_sols or any(not jts for jts in ik_sols):
                    # if verbose:
                    #     print('no joint solution found at {}'.format(cart_proc))
                    continue
                else:
                    graph.resize(len(ee_poses))
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

                        edge_builder = EdgeBuilder(st_size, end_size, dof)
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

                    # vertically concatenate graphs, no extra edges added
                    if vertical_graph.size == 0 and graph.size > 0:
                        # no graph from above
                        vertical_graph = graph
                    else:
                        vertical_subgraph_cnt += 1
                        concatenate_graph_vertically(vertical_graph, graph)
            except StopIteration:
                if verbose: print('pose gen fn iterator exhausted!')
                break
        if vertical_graph.size > 0:
            graph_list.append(vertical_graph)
            if verbose: print('#{}-{} #{} pose families formed.'.format(cp_id, cart_proc, vertical_subgraph_cnt))
        else:
            if verbose: print('Warning: cart proce #{}-{} does not have any valid joint sols to form rungs!'.format(cp_id, cart_proc))

    world_saver.restore()

    # * horizontally concatenate the graphs
    unified_graph = LadderGraph(dof)
    for g in graph_list:
        unified_graph = append_ladder_graph(unified_graph, g)

    # * DAG solve for the concatenated graph
    dag_search = DAGSearch(unified_graph)
    dag_search.run()
    tot_traj = dag_search.shortest_path()

    # TODO: return trajectory for each processes
    return tot_traj

# def generate_ladder_graph_from_poses(robot, dof, pose_list, collision_fn=lambda x: False, dt=-1):
#     # TODO: lazy collision check
#     # TODO: dt, timing constraint
#     graph = LadderGraph(dof)
#     graph.resize(len(pose_list))

#     # solve ik for each pose, build all rungs (w/o edges)
#     for i, pose in enumerate(pose_list):
#         jt_list = sample_tool_ik(robot, pose, get_all=True)
#         jt_list = [jts for jts in jt_list if jts and not collision_fn(jts)]
#         if not jt_list or all(not jts for jts in jt_list):
#            return None
#         graph.assign_rung(i, jt_list)

#     # build edges
#     for i in range(graph.get_rungs_size()-1):
#         st_id = i
#         end_id = i + 1
#         jt1_list = graph.get_data(st_id)
#         jt2_list = graph.get_data(end_id)
#         st_size = graph.get_rung_vert_size(st_id)
#         end_size = graph.get_rung_vert_size(end_id)
#         edge_builder = EdgeBuilder(st_size, end_size, dof)

#         for k in range(st_size):
#             st_id = k * dof
#             for j in range(end_size):
#                 end_id = j * dof
#                 edge_builder.consider(jt1_list[st_id : st_id+dof], jt2_list[end_id : end_id+dof], j)
#             edge_builder.next(k)

#         edges = edge_builder.result
#         if not edge_builder.has_edges and DEBUG:
#             print('no edges!')

#         graph.assign_edges(i, edges)
#     return graph
