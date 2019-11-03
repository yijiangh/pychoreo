
################################################
# result postprocessing utils

def divide_list_chunks(list, size_list):
    assert(sum(size_list) >= len(list))
    if sum(size_list) < len(list):
        size_list.append(len(list) - sum(size_list))
    for j in range(len(size_list)):
        cur_id = sum(size_list[0:j])
        yield list[cur_id:cur_id+size_list[j]]

# temp... really ugly...
def divide_nested_list_chunks(list, size_lists):
    # assert(sum(size_list) >= len(list))
    cur_id = 0
    output_list = []
    for sl in size_lists:
        sub_list = {}
        for proc_name, jt_num in sl.items():
            sub_list[proc_name] = list[cur_id:cur_id+jt_num]
            cur_id += jt_num
        output_list.append(sub_list)
    return output_list

################################################
# ladder graph utils

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
    n_st_vert = int(len(a_rung.data) / dof)
    n_end_vert = int(len(b_rung.data) / dof)

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
    assert graph_above.size == graph_below.size, 'must have same amount of rungs!'# same number of rungs
    num_rungs = graph_above.size
    for i in range(num_rungs):
        rung_above = graph_above.get_rung(i)
        above_jts = graph_above.get_rung(i).data
        below_jts = graph_below.get_rung(i).data
        above_jts.extend(below_jts)
        if i != num_rungs - 1:
            # shifting target vert id in below_edges
            next_above_rung_size = graph_above.get_rung_vert_size(i + 1)
            below_edges = graph_below.get_edges(i)
            for v_out_edges in below_edges:
                e_copy = deepcopy(v_out_edges)
                for v_out_e in e_copy:
                    v_out_e.idx += next_above_rung_size
                rung_above.edges.append(e_copy)
    return graph_above
