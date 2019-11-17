import os
import json
import random
import numpy as np
import pytest
from itertools import chain

from pybullet_planning import INF
from pybullet_planning import load_pybullet, connect, wait_for_user, LockRenderer, has_gui, WorldSaver, HideOutput, \
    reset_simulation, disconnect
from pybullet_planning import interpolate_poses, multiply, unit_pose, get_relative_pose
from pybullet_planning import interval_generator, sample_tool_ik
from pybullet_planning import Pose, Point, Euler, unit_pose
from pybullet_planning import joints_from_names, link_from_name, has_link, get_collision_fn, get_disabled_collisions, \
    draw_pose, set_pose, set_joint_positions, dump_body, dump_world, get_body_body_disabled_collisions

from pychoreo.process_model.cartesian_process import CartesianProcess, CartesianSubProcess
from pychoreo.process_model.trajectory import Trajectory, MotionTrajectory
from pychoreo.process_model.gen_fn import CartesianPoseGenFn
from pychoreo.utils.stream_utils import get_random_direction_generator, get_enumeration_pose_generator
from pychoreo.cartesian_planner.ladder_graph_interface import solve_ladder_graph_from_cartesian_processes

import pychoreo_examples
from pychoreo_examples.extrusion.parsing import load_extrusion, create_elements_bodies
from pychoreo_examples.extrusion.visualization import set_extrusion_camera, draw_extrusion_sequence, display_trajectories
from pychoreo_examples.extrusion.stream import extrusion_ee_pose_gen_fn
from pychoreo_examples.extrusion.utils import max_valence_extrusion_direction_routing, add_collision_fns_from_seq
from pychoreo_examples.extrusion.trajectory import PrintTrajectory, PrintBufferTrajectory
from pychoreo_examples.extrusion.transition_planner import solve_transition_between_extrusion_processes

import ikfast_kuka_kr6_r900

from compas.robots import RobotModel
from compas_fab.robots import Robot as RobotClass
from compas_fab.robots import RobotSemantics

@pytest.fixture
def problem():
    return 'four-frame'
    # return 'simple_frame'

def get_problem_path(problem):
    EXTRUSION_DIRECTORY = pychoreo_examples.get_data('assembly_instances/extrusion')
    EXTRUSION_FILENAMES = {
        'topopt-100': 'topopt-100_S1_03-14-2019_w_layer.json',
        'voronoi': 'voronoi_S1_03-14-2019_w_layer.json',
        'four-frame': 'four-frame.json',
        'simple_frame': 'simple_frame.json',
    }
    EXTRUSION_SEQ_FILENAMES = {
        'four-frame': 'four-frame_solution_regression-z.json',
        'simple_frame': 'simple_frame_solution_regression-z.json',
    }
    here = os.path.dirname(__file__)
    assert problem in EXTRUSION_FILENAMES and problem in EXTRUSION_SEQ_FILENAMES
    return os.path.join(EXTRUSION_DIRECTORY, EXTRUSION_FILENAMES[problem]), \
           os.path.join(here, 'test_data', EXTRUSION_SEQ_FILENAMES[problem])

def get_robot_data():
    URDF_PATH = 'models/kuka_kr6_r900/urdf/kuka_kr6_r900_extrusion.urdf'
    robot_urdf = pychoreo_examples.get_data(URDF_PATH)
    SRDF_PATH = 'models/kuka_kr6_r900/srdf/kuka_kr6_r900_extrusion.srdf'
    robot_srdf = pychoreo_examples.get_data(SRDF_PATH)

    WORKSPACE_URDF_PATH = 'models/kuka_kr6_r900/urdf/mit_3-412_workspace.urdf'
    workspace_urdf = pychoreo_examples.get_data(WORKSPACE_URDF_PATH)
    WORKSPACE_SRDF_PATH = 'models/kuka_kr6_r900/srdf/mit_3-412_workspace.srdf'
    workspace_srdf = pychoreo_examples.get_data(WORKSPACE_SRDF_PATH)

    move_group = 'manipulator_ee'

    robot_model = RobotModel.from_urdf_file(robot_urdf)
    robot_semantics = RobotSemantics.from_srdf_file(robot_srdf, robot_model)
    robot = RobotClass(robot_model, semantics=robot_semantics)

    base_link_name = robot.get_base_link_name(group=move_group)
    ee_link_name = robot.get_end_effector_link_name(group=move_group)
    ik_joint_names = robot.get_configurable_joint_names(group=move_group)
    disabled_self_collision_link_names = robot_semantics.get_disabled_collisions()
    tool_root_link_name = 'eef_base_link' # TODO: should be derived from SRDF as well

    workspace_model = RobotModel.from_urdf_file(workspace_urdf)
    workspace_semantics = RobotSemantics.from_srdf_file(workspace_srdf, workspace_model)
    workspace_robot_disabled_link_names = workspace_semantics.get_disabled_collisions()

    return (robot_urdf, base_link_name, tool_root_link_name, ee_link_name, ik_joint_names, disabled_self_collision_link_names), \
           (workspace_urdf, workspace_robot_disabled_link_names)

def load_extrusion_end_effector():
    with HideOutput():
        ee = load_pybullet(pychoreo_examples.get_data('models/kuka_kr6_r900/urdf/extrusion_end_effector.urdf'))
    return ee

def build_extrusion_cartesian_process(elements, node_points, robot, ik_fn, ik_joint_names, base_link_name, tool_from_root=None, viz_step=False):
    def get_sample_ik_fn(robot, ik_fn, ik_joint_names, base_link_name, tool_from_root=None):
        def sample_ik_fn(world_from_tcp):
            if tool_from_root:
                world_from_tcp = multiply(world_from_tcp, tool_from_root)
            return sample_tool_ik(ik_fn, robot, ik_joint_names, base_link_name, world_from_tcp, get_all=True)
        return sample_ik_fn

    # ik generation function stays the same for all cartesian processes
    sample_ik_fn = get_sample_ik_fn(robot, ik_fn, ik_joint_names, base_link_name, tool_from_root)

    # load EE body, for debugging purpose
    ee_body = load_extrusion_end_effector()
    ik_joints = joints_from_names(robot, ik_joint_names)

    cart_traj_dict = {}
    for element in elements:
        process_name = 'extrusion-E{}'.format(element)
        element = tuple(element)
        n1, n2 = element
        path_pts = [node_points[n1], node_points[n2]]

        # an example for EE pose random generation, yaw (rotation around the direction axis) is set to 0
        random_dir_gen = get_random_direction_generator()
        ee_pose_gen_fn = CartesianPoseGenFn(path_pts, extrusion_ee_pose_gen_fn(path_pts, random_dir_gen, interpolate_poses, approach_distance=0.01, pos_step_size=0.003))

        # build three sub-processes: approach, extrusion, retreat
        extrusion_sub_procs = [CartesianSubProcess(sub_process_name='approach-extrude'),
                               CartesianSubProcess(sub_process_name='extrude'),
                               CartesianSubProcess(sub_process_name='extrude-retreat')]

        cart_process = CartesianProcess(process_name=process_name,
            robot=robot, ik_joint_names=ik_joint_names,
            sub_process_list=extrusion_sub_procs,
            ee_pose_gen_fn=ee_pose_gen_fn, sample_ik_fn=sample_ik_fn,
            element_identifier=element)

        ee_poses = cart_process.sample_ee_poses()
        if viz_step:
            for sp_id, sp in enumerate(ee_poses):
                print('E #{} - sub process #{}'.format(element, sp_id))
                for ee_p in sp:
                    yaw = random.uniform(-np.pi, np.pi)
                    ee_p = multiply(ee_p, Pose(euler=Euler(yaw=yaw)), tool_from_root)
                    set_pose(ee_body, ee_p)
                    if has_gui(): wait_for_user()

        # this should raise an not implemented error since we haven't specify the collision function yet
        for sp in cart_process.sub_process_list:
            with pytest.raises(NotImplementedError):
                conf = [0] * 6
                sp.collision_fn(conf)

        ik_sols = cart_process.get_ik_sols(ee_poses, check_collision=False)
        if viz_step:
            for sp_id, sp_jt_sols in ik_sols.items():
                for jt_sols in sp_jt_sols:
                    for jts in jt_sols:
                        set_joint_positions(robot, ik_joints, jts)
                        if has_gui(): wait_for_user()

        cart_traj_dict[element] = cart_process
    return cart_traj_dict

@pytest.mark.extrusion
def test_extrusion_ladder_graph(problem, viewer):
    # * create robot and pb environment
    (robot_urdf, base_link_name, tool_root_link_name, ee_link_name, ik_joint_names, disabled_self_collision_link_names), \
        (workspace_urdf, workspace_robot_disabled_link_names) = get_robot_data()
    connect(use_gui=viewer)
    with HideOutput():
        robot = load_pybullet(robot_urdf, fixed_base=True)
        workspace = load_pybullet(workspace_urdf, fixed_base=True)
    ik_fn = ikfast_kuka_kr6_r900.get_ik
    ik_joints = joints_from_names(robot, ik_joint_names)

    # * printout bodies in the scene information
    # dump_world()
    # if has_gui() : wait_for_user()

    initial_conf = [0.08, -1.57, 1.74, 0.08, 0.17, -0.08]
    set_joint_positions(robot, ik_joints, initial_conf)

    # * get tool TCP from flange (ee_link) transformation
    # this is for the robot that has end effector specified in its URDF
    root_link = link_from_name(robot, tool_root_link_name)
    tool_link = link_from_name(robot, ee_link_name)
    tool_from_root = get_relative_pose(robot, root_link, tool_link)

    # * get problem & pre-computed json file paths
    file_path, seq_file_path = get_problem_path(problem)

    # * load shape (get nodal positions)
    elements, node_points, ground_nodes = load_extrusion(file_path)
    assert all(isinstance(e, tuple) and len(e) == 2 for e in elements)
    assert all(isinstance(pt, np.ndarray) and len(pt) == 3 for pt in node_points)
    assert all(isinstance(gn, int) for gn in ground_nodes)

    # * create element bodies (collision geometries)
    with LockRenderer():
        draw_pose(unit_pose(), length=1.)
        element_bodies = dict(zip(elements,
            create_elements_bodies(node_points, elements, radius=0.001, shrink=0.008)))
        assert all(isinstance(e_body, int) for e_body in element_bodies.values())
        set_extrusion_camera(node_points)

    # * create cartesian processes without a sequence being given, with random pose generators
    cart_process_dict = build_extrusion_cartesian_process(elements, node_points, robot, ik_fn, ik_joint_names,
        base_link_name, tool_from_root, viz_step=False)

    # * load precomputed sequence
    with open(seq_file_path, 'r') as f:
        seq_data = json.loads(f.read())
    element_sequence = [tuple(e) for e in seq_data['plan']]
    assert all(isinstance(e, tuple) and len(e) == 2 for e in element_sequence)

    # TODO: not implemented yet
    reverse_flags = max_valence_extrusion_direction_routing(element_sequence, elements, ground_nodes)
    assert isinstance(reverse_flags, list)
    assert all(isinstance(flag, bool) for flag in reverse_flags)

    sample_time = 1
    roll_disc = 10
    pitch_disc = 10
    yaw_sample_size = 5
    linear_step_size = 0.003 # mm
    domain_size = roll_disc * pitch_disc

    # * construct ignored body-body links for collision checking
    # in this case, including self-collision between links of the robot
    disabled_self_collisions = get_disabled_collisions(robot, disabled_self_collision_link_names)
    # and links between the robot and the workspace (e.g. robot_base_link to base_plate)
    extra_disabled_collisions = get_body_body_disabled_collisions(robot, workspace, workspace_robot_disabled_link_names)

    def get_ee_pose_map_fn(roll_disc, pitch_disc):
        def ee_pose_map_fn(id, yaw=None):
            j = id % roll_disc
            i = (id - j) / pitch_disc
            roll = -np.pi + i*(2*np.pi/roll_disc)
            pitch = -np.pi + j*(2*np.pi/pitch_disc)
            yaw = random.uniform(-np.pi, np.pi) if yaw is None else yaw
            return Pose(euler=Euler(roll=roll, pitch=pitch, yaw=yaw))
        return ee_pose_map_fn

    with WorldSaver():
        ee_body = load_extrusion_end_effector()
        ee_pose_map_fn = get_ee_pose_map_fn(roll_disc, pitch_disc)

        # * building collision function based on the given sequence
        with LockRenderer(False):
            cart_process_seq, e_fmaps = add_collision_fns_from_seq(
                robot, ik_joints, cart_process_dict,
                element_sequence, element_bodies,
                domain_size, ee_pose_map_fn, ee_body,
                sample_time=sample_time, yaw_sample_size=yaw_sample_size, linear_step_size=linear_step_size, tool_from_root=tool_from_root,
                self_collisions=True, disabled_collisions=disabled_self_collisions,
                obstacles=[workspace], extra_disabled_collisions=extra_disabled_collisions, verbose=True)

        assert isinstance(cart_process_seq, list)
        assert all(isinstance(cp, CartesianProcess) for cp in cart_process_seq)
        assert all([cart.element_identifier == e for cart, e in zip(cart_process_seq, element_sequence)])

        # * draw the pruned EE direction set
        if has_gui():
            # just move the element bodies and ee_body away to clear the visualized scene
            set_pose(ee_body, unit_pose())
            for e_body in element_bodies.values(): set_pose(e_body, unit_pose())
            draw_extrusion_sequence(node_points, element_bodies, element_sequence, e_fmaps, ee_pose_map_fn=ee_pose_map_fn,
                                    line_width=5, direction_len=0.005, time_step=0.01)

    viz_inspect = False
    with LockRenderer(not viz_inspect):
        cart_process_seq = solve_ladder_graph_from_cartesian_processes(cart_process_seq, verbose=True, warning_pause=False, viz_inspect=viz_inspect, check_collision=True)
        assert all(isinstance(cp, CartesianProcess) for cp in cart_process_seq)

        # TODO: we can do reverse processing here, instead of inside add_collision_fns_from_seq?

        # * extract trajectory from CartProcesses
        print_trajs = [[] for _ in range(len(cart_process_seq))]
        for cp_id, cp in enumerate(cart_process_seq):
            for sp_id, sp in enumerate(cp.sub_process_list):
                # assert sp.trajectory, '{}-{} does not have a Cartesian plan found!'.format(cp, sp)
                if sp_id == 0:
                    print_trajs[cp_id].append(PrintBufferTrajectory.from_trajectory(sp.trajectory, cp.element_identifier, reverse_flags[cp_id], tag='approach'))
                elif sp_id == 1:
                    print_trajs[cp_id].append(PrintTrajectory.from_trajectory(sp.trajectory, cp.element_identifier, reverse_flags[cp_id]))
                else:
                    print_trajs[cp_id].append(PrintBufferTrajectory.from_trajectory(sp.trajectory, cp.element_identifier, reverse_flags[cp_id], tag='retreat'))

    # TODO get rid of this when transition planning is done
    full_trajs = print_trajs

    # # * transition motion planning between extrusions
    return2idle = True
    transition_traj = solve_transition_between_extrusion_processes(robot, ik_joints, print_trajs, element_bodies, initial_conf,
                                                                   disabled_collisions=disabled_self_collisions,
                                                                   obstacles=[], return2idle=return2idle)
    assert all(isinstance(tt, MotionTrajectory) for tt in transition_traj)
    if return2idle:
        assert len(transition_traj)-1 == len(print_trajs)
    else:
        assert len(transition_traj) == len(print_trajs)

    # # * weave the Cartesian and transition processses together
    for cp_id, print_trajs in enumerate(full_trajs):
        print_trajs.insert(0, transition_traj[cp_id])
    if return2idle:
        full_trajs[-1].append(transition_traj[-1])

    # * disconnect and close pybullet engine used for planning, visualizing trajectories will start a new one
    reset_simulation()
    disconnect()

    # visualize plan
    if viewer:
        display_trajectories(robot_urdf, ik_joint_names, ee_link_name, node_points, ground_nodes, full_trajs,
                             workspace_urdf=workspace_urdf, animate=True, cart_time_step=0.02, tr_time_step=0.05)
