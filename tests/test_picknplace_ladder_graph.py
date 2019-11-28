import os
import json
import random
import numpy as np
import pytest
from itertools import chain

from compas_fab.backends.pybullet import convert_mesh_to_pybullet_body
from compas_fab.assembly.datastructures import Assembly, UnitGeometry, Grasp

from pybullet_planning import load_pybullet, connect, wait_for_user, LockRenderer, has_gui, WorldSaver, HideOutput, \
    reset_simulation, disconnect, set_camera_pose, remove_handles
from pybullet_planning import multiply, unit_pose, get_relative_pose, invert
from pybullet_planning import interval_generator, sample_tool_ik
from pybullet_planning import Pose, Point, Euler, unit_pose, Attachment
from pybullet_planning import joints_from_names, link_from_name, has_link, get_collision_fn, get_disabled_collisions, \
    draw_pose, set_pose, set_joint_positions, dump_world, create_obj, get_body_body_disabled_collisions, get_link_pose, \
    create_attachment, set_pose, get_pose, link_from_name, BASE_LINK

from pychoreo.process_model.cartesian_process import CartesianProcess
from pychoreo.process_model.trajectory import Trajectory, MotionTrajectory
from pychoreo.utils.stream_utils import get_random_direction_generator, get_enumeration_pose_generator
from pychoreo.utils.parsing_utils import export_trajectory
from pychoreo.cartesian_planner.ladder_graph_interface import solve_ladder_graph_from_cartesian_process_list
from pychoreo.cartesian_planner.sparse_ladder_graph import SparseLadderGraph

from pychoreo_examples.picknplace.stream import build_picknplace_cartesian_process_seq
from pychoreo_examples.picknplace.trajectory import PicknPlaceBufferTrajectory
from pychoreo_examples.picknplace.visualization import display_picknplace_trajectories
from pychoreo_examples.picknplace.parsing import parse_saved_trajectory
from pychoreo_examples.picknplace.transition_planner import solve_transition_between_picknplace_processes

import ikfast_ur5

def load_end_effector(ee_urdf_path):
    with HideOutput():
        ee = load_pybullet(ee_urdf_path)
    return ee

@pytest.mark.pnp
# @pytest.mark.parametrize('solve_method', [('sparse_ladder_graph')])
# @pytest.mark.parametrize('solve_method', [('ladder_graph')])
@pytest.mark.parametrize('solve_method', [('ladder_graph'), ('sparse_ladder_graph')])
def test_picknplace_ladder_graph(viewer, picknplace_problem_path, picknplace_robot_data,
    picknplace_end_effector, picknplace_tcp_def, solve_method):

    step_num = 5
    sample_time = 5
    sparse_time_out = 2
    jt_res = 0.1

    # * create robot and pb environment
    (robot_urdf, base_link_name, tool_root_link_name, ee_link_name, ik_joint_names, disabled_self_collision_link_names), \
    (workspace_urdf, workspace_robot_disabled_link_names) = picknplace_robot_data
    connect(use_gui=viewer)

    # * adjust camera pose (optional)
    camera_base_pt = (0,0,0)
    camera_pt = np.array(camera_base_pt) + np.array([1, 0, 0.5])
    set_camera_pose(tuple(camera_pt), camera_base_pt)

    with HideOutput():
        # * pybullet can handle ROS-package path URDF automatically now (ver 2.5.7)!
        robot = load_pybullet(robot_urdf, fixed_base=True)
        workspace = load_pybullet(workspace_urdf, fixed_base=True)
    ik_fn = ikfast_ur5.get_ik

    # * set robot idle configuration
    ik_joints = joints_from_names(robot, ik_joint_names)
    robot_start_conf = [0,-1.65715,1.71108,-1.62348,0,0]
    set_joint_positions(robot, ik_joints, robot_start_conf)

    # * create tool and tool TCP from flange (tool0) transformation
    root_link = link_from_name(robot, tool_root_link_name)
    # create end effector body
    ee_body = load_end_effector(picknplace_end_effector)
    ee_attach = Attachment(robot, root_link, unit_pose(), ee_body)
    # set up TCP transformation, just a renaming here
    root_from_tcp = picknplace_tcp_def
    if has_gui() :
        # draw_tcp pose
        ee_attach.assign()
        ee_link_pose = get_pose(ee_attach.child)
        draw_pose(multiply(ee_link_pose, root_from_tcp))

    # * specify ik fn wrapper
    ik_fn = ikfast_ur5.get_ik
    def get_sample_ik_fn(robot, ik_fn, ik_joint_names, base_link_name, tool_from_root=None):
        def sample_ik_fn(world_from_tcp):
            if tool_from_root:
                world_from_tcp = multiply(world_from_tcp, tool_from_root)
            return sample_tool_ik(ik_fn, robot, ik_joint_names, base_link_name, world_from_tcp, get_all=True, sampled=[0])
        return sample_ik_fn
    # ik generation function stays the same for all cartesian processes
    sample_ik_fn = get_sample_ik_fn(robot, ik_fn, ik_joint_names, base_link_name, invert(root_from_tcp))

    # * get problem & pre-computed json file paths
    pkg_json_path, _ = picknplace_problem_path

    # * load shape & collision data
    with open(pkg_json_path, 'r') as f:
        json_data = json.loads(f.read())
    assembly = Assembly.from_package(json_data)
    elements = assembly.elements
    for element in elements.values():
        for unit_geo in element.unit_geometries:
            unit_geo.rescale(1e-3)
    # static_obstacles = []
    # for ug in assembly.static_obstacle_geometries.values():
    #     static_obstacles.extend(ug.mesh)

    # * load precomputed sequence / use assigned sequence
    # element_seq = elements.keys()
    element_seq = [0, 2, 3]
    print('sequence: ', element_seq)

    # visualize goal pose
    if has_gui():
        viz_len = 0.003
        with WorldSaver():
            for e_id in element_seq:
                element = elements[e_id]
                with LockRenderer():
                    print('e_id #{} : {}'.format(e_id, element))
                    for unit_geo in element.unit_geometries:
                        for pb_geo in unit_geo.pybullet_bodies:
                            set_pose(pb_geo, random.choice(unit_geo.get_goal_frames(get_pb_pose=True)))
                # print('---------')
                # wait_for_user()
        wait_for_user()

    # * construct ignored body-body links for collision checking
    # in this case, including self-collision between links of the robot
    disabled_self_collisions = get_disabled_collisions(robot, disabled_self_collision_link_names)
    # and links between the robot and the workspace (e.g. robot_base_link to base_plate)
    extra_disabled_collisions = get_body_body_disabled_collisions(robot, workspace, workspace_robot_disabled_link_names)
    extra_disabled_collisions.add(
        ((robot, link_from_name(robot, 'wrist_3_link')), (ee_body, BASE_LINK))
        )

    # * create cartesian processes without a sequence being given, with random pose generators
    cart_process_seq = build_picknplace_cartesian_process_seq(
        element_seq, elements,
        robot, ik_joint_names, root_link, sample_ik_fn,
        num_steps=step_num, ee_attachs=[ee_attach],
        self_collisions=True, disabled_collisions=disabled_self_collisions,
        obstacles=[workspace],extra_disabled_collisions=extra_disabled_collisions,
        tool_from_root=invert(root_from_tcp), viz_step=False, pick_from_same_rack=True)

    # specifically for UR5, because of its wide joint range, we need to apply joint value snapping
    for cp in cart_process_seq:
        cp.target_conf = robot_start_conf

    viz_inspect = False
    with LockRenderer(not viz_inspect):
        if solve_method == 'ladder_graph':
            print('\n'+'#' * 10)
            print('Solving with the vanilla ladder graph search algorithm.')
            cart_process_seq = solve_ladder_graph_from_cartesian_process_list(cart_process_seq,
                verbose=True, warning_pause=False, viz_inspect=viz_inspect, check_collision=True, start_conf=robot_start_conf)
        elif solve_method == 'sparse_ladder_graph':
            print('\n'+'#' * 10)
            print('Solving with the sparse ladder graph search algorithm.')
            sparse_graph = SparseLadderGraph(cart_process_seq)
            sparse_graph.find_sparse_path(verbose=True, vert_timeout=sample_time, sparse_sample_timeout=sparse_time_out)
            cart_process_seq = sparse_graph.extract_solution(verbose=True, start_conf=robot_start_conf)
        else:
            raise ValueError('Invalid solve method!')
        assert all(isinstance(cp, CartesianProcess) for cp in cart_process_seq)

        pnp_trajs = [[] for _ in range(len(cart_process_seq))]
        for cp_id, cp in enumerate(cart_process_seq):
            element_attachs = []
            for sp_id, sp in enumerate(cp.sub_process_list):
                assert sp.trajectory, '{}-{} does not have a Cartesian plan found!'.format(cp, sp)
                # ! reverse engineer the grasp pose
                if sp.trajectory.tag == 'pick_retreat':
                    unit_geo = elements[sp.trajectory.element_id].unit_geometries[0]
                    e_bodies = unit_geo.pybullet_bodies
                    for e_body in e_bodies:
                        set_pose(e_body, unit_geo.get_initial_frames(get_pb_pose=True)[0])
                        set_joint_positions(sp.trajectory.robot, sp.trajectory.joints, sp.trajectory.traj_path[0])
                        element_attachs.append(create_attachment(sp.trajectory.robot, root_link, e_body))

                if sp.trajectory.tag == 'pick_retreat' or sp.trajectory.tag == 'place_approach':
                    sp.trajectory.attachments= element_attachs
                pnp_trajs[cp_id].append(sp.trajectory)
    full_trajs = pnp_trajs

    # * transition motion planning between extrusions
    return2idle = True
    transition_traj = solve_transition_between_picknplace_processes(pnp_trajs, elements, robot_start_conf,
                                                                    disabled_collisions=disabled_self_collisions,
                                                                    extra_disabled_collisions=extra_disabled_collisions,
                                                                    obstacles=[workspace], return2idle=return2idle,
                                                                    resolutions=[jt_res]*len(ik_joints))

    # * weave the Cartesian and transition processses together
    for cp_id, print_trajs in enumerate(full_trajs):
        print_trajs.insert(0, transition_traj[cp_id][0])
        print_trajs.insert(3, transition_traj[cp_id][1])
    if return2idle:
        full_trajs[-1].append(transition_traj[-1][-1])

    here = os.path.dirname(__file__)
    save_dir = os.path.join(here, 'results')
    export_trajectory(save_dir, full_trajs, ee_link_name, indent=None, shape_file_path=pkg_json_path,
        include_robot_data=True, include_link_path=False)

    # * disconnect and close pybullet engine used for planning, visualizing trajectories will start a new one
    reset_simulation()
    disconnect()

    if viewer:
        cart_time_step = None
        tr_time_step = None
        display_picknplace_trajectories(robot_urdf, ik_joint_names,
                                        pkg_json_path, full_trajs, tool_root_link_name,
                                        ee_urdf=picknplace_end_effector, workspace_urdf=workspace_urdf, animate=True,
                                        cart_time_step=cart_time_step, tr_time_step=tr_time_step)

@pytest.mark.pnp_viz
def test_parse_and_visualize_results(viewer, picknplace_problem_path, picknplace_robot_data,
    picknplace_end_effector): #, picknplace_tcp_def

    # * create robot and pb environment
    (robot_urdf, base_link_name, tool_root_link_name, ee_link_name, ik_joint_names, disabled_self_collision_link_names), \
    (workspace_urdf, workspace_robot_disabled_link_names) = picknplace_robot_data

    # * get problem & pre-computed json file paths
    pkg_json_path, result_file_name = picknplace_problem_path

    # * parse saved trajectory results
    here = os.path.dirname(__file__)
    save_file_path = os.path.join(here, 'results', result_file_name)

    # parse without connect
    with pytest.warns(UserWarning, match='Pybullet environment not connected*'):
        full_trajs = parse_saved_trajectory(save_file_path)

    # parse with connect but robot body not added
    connect(use_gui=False)
    with pytest.raises(ValueError):
        full_trajs = parse_saved_trajectory(save_file_path)
    disconnect()

    # parse with connect
    connect(use_gui=False)
    with HideOutput():
        robot = load_pybullet(robot_urdf, fixed_base=True)
    # with pytest.warns(UserWarning, match='Cannot find body with name*'):
    full_trajs = parse_saved_trajectory(save_file_path)
    disconnect()

    # visualize plan
    if viewer:
        cart_time_step = None
        tr_time_step = None
        display_picknplace_trajectories(robot_urdf, ik_joint_names,
                                        pkg_json_path, full_trajs, tool_root_link_name,
                                        ee_urdf=picknplace_end_effector, workspace_urdf=workspace_urdf, animate=True,
                                        cart_time_step=cart_time_step, tr_time_step=tr_time_step)
