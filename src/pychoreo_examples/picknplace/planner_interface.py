
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


def quick_check_place_feasibility(robot, ik_joint_names, base_link_name, ee_link_name, ik_fn,
    unit_geo,
    num_cart_steps=10,
    static_obstacles=[], self_collisions=True,
    mount_link_from_tcp_pose=None, ee_attachs=[], viz=False,
    st_conf=[], disabled_collision_link_names=[], diagnosis=False, viz_time_gap=1, draw_pose_size=0.02):

    ik_joints = joints_from_names(robot, ik_joint_names)
    tool_link = link_from_name(robot, ee_link_name)
    disabled_collision_links = [(link_from_name(robot, pair[0]), link_from_name(robot, pair[1])) \
         for pair in disabled_collision_link_names]

    # generate path pts
    for place_grasp, goal_pose in zip(unit_geo.place_grasps, unit_geo.goal_pb_poses):
        pose_handle = [] # visualization handle

        def make_assembly_poses(obj_pose, grasp_poses):
            return [multiply(obj_pose, g_pose) for g_pose in grasp_poses]

        # for e_body in unit_geo.pybullet_bodies:
        #     set_pose(e_body, unit_geo.initial_pb_pose)

        grasp_pose_seq = [place_grasp.object_from_approach_pb_pose,
                          place_grasp.object_from_attach_pb_pose,
                          place_grasp.object_from_retreat_pb_pose]
        world_from_place_poses = make_assembly_poses(goal_pose, grasp_pose_seq)

        # pose_handle.append(draw_pose(goal_pose, length=draw_pose_size))

        # approach2attach_place_old = interpolate_cartesian_poses(world_from_place_poses[0], world_from_place_poses[1],
        # disc_len, mount_link_from_tcp=mount_link_from_tcp_pose)
        approach2attach_place_tcp = list(interpolate_poses_by_num(world_from_place_poses[0], world_from_place_poses[1], \
            num_steps=num_cart_steps))
        if has_gui() and viz:
            for p_tmp in approach2attach_place_tcp:
                pose_handle.append(draw_pose(p_tmp, length=draw_pose_size))
        if mount_link_from_tcp_pose:
            approach2attach_place = [multiply(world_from_tcp, invert(mount_link_from_tcp_pose)) \
                for world_from_tcp in approach2attach_place_tcp]

        attach2retreat_place = list(interpolate_poses_by_num(world_from_place_poses[1], world_from_place_poses[2], \
            num_steps=num_cart_steps))
        if has_gui() and viz:
            for p_tmp in attach2retreat_place:
                pose_handle.append(draw_pose(p_tmp, length=draw_pose_size))
        if mount_link_from_tcp_pose:
            attach2retreat_place = [multiply(world_from_tcp, invert(mount_link_from_tcp_pose)) \
                for world_from_tcp in attach2retreat_place]

        picknplace_poses = approach2attach_place + attach2retreat_place

        if mount_link_from_tcp_pose:
            attach_from_object = multiply(mount_link_from_tcp_pose, invert(place_grasp.object_from_attach_pb_pose))
        else:
            attach_from_object = invert(place_grasp.object_from_attach_pb_pose)

        # generating the element attachment
        temp_jt_list = sample_tool_ik(ik_fn, robot, ik_joint_names, base_link_name, approach2attach_place[-1], get_all=True)
        if not temp_jt_list:
            if has_gui() and viz:
                for l in [line for pose in pose_handle for line in pose]:
                    remove_debug(l)
            continue
        set_joint_positions(robot, ik_joints, temp_jt_list[0])
        for e_body in unit_geo.pybullet_bodies:
            set_pose(e_body, unit_geo.goal_pb_pose)
        attachs = [Attachment(robot, tool_link, attach_from_object, e_body) for e_body in unit_geo.pybullet_bodies]
        if ee_attachs:
            attachs.extend(ee_attachs)

        # approach 2 place
        collision_fn_approach_place = get_collision_fn(robot, ik_joints,
                                        static_obstacles,
                                        attachments=attachs, self_collisions=self_collisions,
                                        disabled_collisions=disabled_collision_links,
                                        diagnosis=diagnosis, viz_last_duration=viz_time_gap)

        ignored_pairs = list(product([ee_attach.child for ee_attach in ee_attachs], unit_geo.pybullet_bodies))
        # place 2 retreat
        collision_fn_retreat_place = get_collision_fn(robot, ik_joints,
                                        static_obstacles,
                                        attachments=ee_attachs, self_collisions=self_collisions,
                                        disabled_collisions=disabled_collision_links,
                                        custom_limits={}, ignored_pairs=ignored_pairs, viz_last_duration=viz_time_gap)

        is_empty = False

        # solve ik for each pose, build all rungs (w/o edges)
        print(len(picknplace_poses))
        assert len(picknplace_poses) == 2*(num_cart_steps + 1)
        for i, pose in enumerate(picknplace_poses):
            jt_list = sample_tool_ik(ik_fn, robot, ik_joint_names, base_link_name, pose, get_all=True)

            if i < num_cart_steps + 1:
                jt_list = [jts for jts in jt_list if jts and not collision_fn_approach_place(jts)]
            else:
                jt_list = [jts for jts in jt_list if jts and not collision_fn_retreat_place(jts)]

            if not jt_list or all(not jts for jts in jt_list):
                print('no joint solution found at brick #{0} path pt #{1} grasp id #{2}'.format( \
                    unit_geo.name, i, place_grasp._grasp_id))
                is_empty = True
                break
            else:
                if has_gui() and viz:
                    # for jt_id, jt in enumerate(jt_list):
                    jt = jt_list[0]
                    set_joint_positions(robot, ik_joints, jt)
                    if i < num_cart_steps + 1:
                        for at in attachs: at.assign()
                        # print('-- ik sol found #{} at element #{} path pt #{} grasp id #{}'.format( \
                        #     jt_id, unit_geo.name, i, place_grasp._grasp_id))
                    wait_for_duration(viz_time_gap)

        if has_gui() and viz:
            wait_for_duration(2)
            for l in [line for pose in pose_handle for line in pose]:
                remove_debug(l)

        if is_empty:
            continue
        else:
            return True
    return False


def generate_ladder_graph_for_picknplace_single_brick(robot, ik_joint_names, base_link_name, ee_link_name, ik_fn,
    unit_geo,
    num_cart_steps=10,
    assembled_element_obstacles=[], unassembled_element_obstacles=[],
    static_obstacles=[], self_collisions=True,
    pick_from_same_rack=False, mount_link_from_tcp_pose=None, ee_attachs=[], viz=False,
    st_conf=[], disabled_collision_link_names=[], verbose=False):

    """ unit_geo : compas_fab.assembly.datastructures.UnitGeometry"""
    # TODO: lazy collision check
    # TODO: dt, timing constraint

    dof = len(ik_joint_names)
    vertical_graph = LadderGraph(dof)
    ik_joints = joints_from_names(robot, ik_joint_names)
    tool_link = link_from_name(robot, ee_link_name)
    disabled_collision_links = [(link_from_name(robot, pair[0]), link_from_name(robot, pair[1])) \
         for pair in disabled_collision_link_names]

    # generate path pts
    g_id_len = len(unit_geo.place_grasps)
    for g_id in range(g_id_len):
        pose_handle = [] # visualization handle
        sub_graph_sizes = {}

        pick_grasp = unit_geo.pick_grasps[g_id]
        place_grasp = unit_geo.place_grasps[g_id]

        def make_assembly_poses(obj_pose, grasp_poses):
            return [multiply(obj_pose, g_pose) for g_pose in grasp_poses]

        for e_body in unit_geo.pybullet_bodies:
            set_pose(e_body, unit_geo.initial_pb_pose)

        # TODO: generalize to an abstract cartesian pose class
        pick_grasp_pose_seq = [pick_grasp.object_from_approach_pb_pose,
                               pick_grasp.object_from_attach_pb_pose,
                               pick_grasp.object_from_retreat_pb_pose]
        place_grasp_pose_seq = [place_grasp.object_from_approach_pb_pose,
                                place_grasp.object_from_attach_pb_pose,
                                place_grasp.object_from_retreat_pb_pose]
        world_from_pick_poses = make_assembly_poses(unit_geo.initial_pb_pose,      pick_grasp_pose_seq)
        world_from_place_poses = make_assembly_poses(unit_geo.goal_pb_poses[g_id], place_grasp_pose_seq)

        # approach2attach_pick = interpolate_cartesian_poses(world_from_pick_poses[0], world_from_pick_poses[1],
        # disc_len, mount_link_from_tcp=mount_link_from_tcp_pose)
        approach2attach_pick = list(interpolate_poses_by_num(world_from_pick_poses[0], world_from_pick_poses[1], \
            num_steps=num_cart_steps))
        if mount_link_from_tcp_pose:
            approach2attach_pick = [multiply(world_from_tcp, invert(mount_link_from_tcp_pose)) \
                for world_from_tcp in approach2attach_pick]
        sub_graph_sizes['pick_approach'] = len(approach2attach_pick)

        # attach2retreat_pick = interpolate_cartesian_poses(world_from_pick_poses[1], world_from_pick_poses[2],
        # disc_len, mount_link_from_tcp=mount_link_from_tcp_pose)
        attach2retreat_pick = list(interpolate_poses_by_num(world_from_pick_poses[1], world_from_pick_poses[2], \
            num_steps=num_cart_steps))
        if mount_link_from_tcp_pose:
            attach2retreat_pick = [multiply(world_from_tcp, invert(mount_link_from_tcp_pose)) \
                for world_from_tcp in attach2retreat_pick]
        sub_graph_sizes['pick_retreat'] = len(attach2retreat_pick)

        # approach2attach_place = interpolate_cartesian_poses(world_from_place_poses[0], world_from_place_poses[1],
        # disc_len, mount_link_from_tcp=mount_link_from_tcp_pose)
        approach2attach_place = list(interpolate_poses_by_num(world_from_place_poses[0], world_from_place_poses[1], \
            num_steps=num_cart_steps))
        if mount_link_from_tcp_pose:
            approach2attach_place = [multiply(world_from_tcp, invert(mount_link_from_tcp_pose)) \
                for world_from_tcp in approach2attach_place]
        sub_graph_sizes['place_approach'] = len(approach2attach_place)

        # attach2retreat_place = interpolate_cartesian_poses(world_from_place_poses[1], world_from_place_poses[2],
        #     disc_len, mount_link_from_tcp=mount_link_from_tcp_pose)
        attach2retreat_place = list(interpolate_poses_by_num(world_from_place_poses[1], world_from_place_poses[2], \
            num_steps=num_cart_steps))
        if mount_link_from_tcp_pose:
            attach2retreat_place = [multiply(world_from_tcp, invert(mount_link_from_tcp_pose)) \
                for world_from_tcp in attach2retreat_place]
        sub_graph_sizes['place_retreat'] = len(attach2retreat_place)

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

        if has_gui() and viz:
            if mount_link_from_tcp_pose:
                picknplace_tcp_viz = [multiply(world_from_mount, mount_link_from_tcp_pose) \
                    for world_from_mount in picknplace_poses]
            for p_tmp in picknplace_tcp_viz:
                pose_handle.append(draw_pose(p_tmp, length=0.02))
            try:
                wait_for_user()
            except:
                wait_for_duration(2)

        collision_fns = []
        if mount_link_from_tcp_pose:
            attach_from_object = multiply(mount_link_from_tcp_pose, invert(pick_grasp.object_from_attach_pb_pose))
        else:
            attach_from_object = invert(pick_grasp.object_from_attach_pb_pose)

        temp_jt_list = sample_tool_ik(ik_fn, robot, ik_joint_names, base_link_name, \
            attach2retreat_pick[0], get_all=True)
        if not temp_jt_list:
            continue
        set_joint_positions(robot, ik_joints, temp_jt_list[0])
        for e_body in unit_geo.pybullet_bodies:
            set_pose(e_body, unit_geo.initial_pb_pose)
        attachs = [Attachment(robot, tool_link, attach_from_object, e_body) for e_body in unit_geo.pybullet_bodies]
        if ee_attachs:
            attachs.extend(ee_attachs)

        ignored_pairs = list(product([ee_attach.child for ee_attach in ee_attachs], unit_geo.pybullet_bodies))
        # appraoch 2 pick
        pick_approach_obstacles = static_obstacles + assembled_element_obstacles if pick_from_same_rack \
            else static_obstacles + assembled_element_obstacles + unassembled_element_obstacles

        collision_fns.append(get_collision_fn(robot, ik_joints, pick_approach_obstacles,
                                                attachments=ee_attachs, self_collisions=self_collisions,
                                                disabled_collisions=disabled_collision_links,
                                                custom_limits={}, ignored_pairs=ignored_pairs))
        # pick 2 retreat
        collision_fns.append(get_collision_fn(robot, ik_joints, pick_approach_obstacles,
                                                attachments=ee_attachs + attachs, self_collisions=self_collisions,
                                                disabled_collisions=disabled_collision_links,
                                                custom_limits={}))
        # approach 2 place
        collision_fns.append(get_collision_fn(robot, ik_joints,
                                                static_obstacles + assembled_element_obstacles + unassembled_element_obstacles,
                                                attachments=ee_attachs + attachs, self_collisions=self_collisions,
                                                disabled_collisions=disabled_collision_links,
                                                custom_limits={}))
        # place 2 retreat
        collision_fns.append(get_collision_fn(robot, ik_joints,
                                                static_obstacles + assembled_element_obstacles + unassembled_element_obstacles,
                                                attachments=ee_attachs, self_collisions=self_collisions,
                                                disabled_collisions=disabled_collision_links,
                                                custom_limits={}, ignored_pairs=ignored_pairs))

        graph = LadderGraph(dof)
        graph.resize(len(picknplace_poses))
        is_empty = False

        # solve ik for each pose, build all rungs (w/o edges)
        for i, pose in enumerate(picknplace_poses):
            # TODO: special sampler for 6+ extra dofs
            # sub_id = 0 -> pick, sub_id = 1 -> place
            sub_id = 0 if process_map[i] < 2 else 1

            jt_list = sample_tool_ik(ik_fn, robot, ik_joint_names, base_link_name, pose, get_all=True)

            if st_conf:
                # print('before snap: ', jt_list)
                joint_limits = [get_joint_limits(robot, pb_joint) for pb_joint in ik_joints]
                jt_list = snap_sols(jt_list, st_conf, joint_limits)
                # print('after snap: ', jt_list)

            if process_map[i] == 3:
                # the object is in its goal pose in place-retreat phase
                for e_body in unit_geo.pybullet_bodies:
                    # TODO: symmetric goal pose
                    set_pose(e_body, unit_geo.goal_pb_poses[g_id])

            jt_list = [jts for jts in jt_list if jts and not collision_fns[process_map[i]](jts)]

            if not jt_list or all(not jts for jts in jt_list):
                if verbose: print('no joint solution found at brick #{0} path pt #{1} grasp id #{2}'.format(\
                                unit_geo.name, i, g_id))
                is_empty = True
                break
            else:
                if has_gui() and viz and process_map[i] >= 2:
                    # only viz placing
                    for jt_id, jt in enumerate(jt_list):
                        set_joint_positions(robot, ik_joints, jt)
                        for ea in ee_attachs: ea.assign()
                        print('-- ik sol found #{} at element #{} path pt #{} grasp id #{}'.format(
                            jt_id, unit_geo.name, i, g_id))
                        try:
                            wait_for_user()
                        except:
                            wait_for_duration(1)

                # print('rung #{0} at brick #{1} grasp id #{2}'.format(i, brick.index, grasp.num))
                graph.assign_rung(i, jt_list)

            if has_gui() and viz:
                for l in [line for pose in pose_handle for line in pose]:
                    remove_debug(l)

        if is_empty:
            # for l in [line for pose in pose_handle for line in pose]:
            #     remove_debug(l)
            continue
        if verbose: print('Found!!! at brick #{} grasp id #{}'.format( \
                        unit_geo.name, g_id))

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
            # if not edge_builder.has_edges and DEBUG:
            #     print('no edges!')

            graph.assign_edges(i, edges)

        if vertical_graph.size == 0:
            vertical_graph = graph
        else:
            concatenate_graph_vertically(vertical_graph, graph)
        # end loop grasps
    return vertical_graph, sub_graph_sizes


def direct_ladder_graph_solve_picknplace(robot, ik_joint_names, base_link_name, tool_link_name, ik_fn,
        unit_geo_dict, element_seq, obstacle_from_name,
        from_seq_id=0, to_seq_id=None,
        disabled_collision_link_names=[], self_collisions=True, pick_from_same_rack=False,
        tcp_transf=None, ee_attachs=[], max_attempts=1, viz=False, st_conf=None, num_cart_steps=10):
    dof = len(get_movable_joints(robot))
    graph_list = []
    static_obstacles = list(obstacle_from_name.values())

    def flatten_unit_geos_bodies(in_dict):
        out_list = []
        for ug in in_dict.values():
            out_list.extend(ug.pybullet_bodies)
        return out_list

    # reset all bodies
    for unit_geo in unit_geo_dict.values():
        for g_body in unit_geo.pybullet_bodies:
            set_pose(g_body, unit_geo.initial_pb_pose)

    to_seq_id = to_seq_id or len(element_seq)-1
    assert 0 <= from_seq_id and from_seq_id < len(element_seq)
    assert from_seq_id <= to_seq_id and to_seq_id < len(element_seq)

    for seq_id in range(0, from_seq_id):
        e_id = element_seq[seq_id]
        for e_body in unit_geo_dict[e_id].pybullet_bodies:
            set_pose(e_body, unit_geo_dict[e_id].goal_pb_pose)

    graph_sizes = []
    for seq_id in range(from_seq_id, to_seq_id + 1):
        e_id = element_seq[seq_id]
        unit_geo = unit_geo_dict[e_id]
        solved = False
        assembled_elements = flatten_unit_geos_bodies({element_seq[i] : unit_geo_dict[element_seq[i]] for i in range(0, seq_id)})
        unassembled_elements = flatten_unit_geos_bodies({element_seq[i] : unit_geo_dict[element_seq[i]] for i in range(seq_id+1, to_seq_id+1)})
        for run_iter in range(max_attempts):
            graph, sub_graph_sizes = \
            generate_ladder_graph_for_picknplace_single_brick(robot, ik_joint_names, base_link_name, tool_link_name, ik_fn,
                unit_geo,
                pick_from_same_rack=pick_from_same_rack,
                static_obstacles=static_obstacles,
                assembled_element_obstacles=assembled_elements,
                unassembled_element_obstacles=unassembled_elements,
                self_collisions=self_collisions,
                num_cart_steps=num_cart_steps,
                disabled_collision_link_names=disabled_collision_link_names,
                ee_attachs=ee_attachs, mount_link_from_tcp_pose=tcp_transf, viz=viz, st_conf=st_conf)

            if graph.size > 0:
                graph_list.append(graph)
                graph_sizes.append(sub_graph_sizes)
                solved = True
                break
            else:
                print('graph empty at brick #{} at seq #{}, rerun #{}'.format(e_id, seq_id, run_iter))

        if not solved:
            print('NOT SOLVED! seq #{}, brick#{} after {} attempts'.format(seq_id, e_id, max_attempts))
            assert False, 'NOT SOLVED! seq #{}, brick#{} after {} attempts'.format(seq_id, e_id, max_attempts)
            # if has_gui():
            #     wait_for_user()
            # else:
            #     input()

        for e_body in unit_geo.pybullet_bodies:
            # TODO: symmetric goal pose
            set_pose(e_body, unit_geo.goal_pb_pose)

    unified_graph = LadderGraph(dof)
    for g in graph_list:
        unified_graph = append_ladder_graph(unified_graph, g)

    dag_search = DAGSearch(unified_graph)
    dag_search.run()
    tot_traj = dag_search.shortest_path()

    return tot_traj, graph_sizes
