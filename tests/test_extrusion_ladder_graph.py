import os
import json
import random
import numpy as np
import pytest

from pybullet_planning import load_pybullet, connect, wait_for_user, LockRenderer, has_gui, WorldSaver, HideOutput
from pybullet_planning import interpolate_poses, multiply, unit_pose, get_relative_pose
from pybullet_planning import interval_generator, sample_tool_ik
from pybullet_planning import Pose, Point, Euler, unit_pose
from pybullet_planning import joints_from_names, link_from_name, has_link, get_collision_fn, get_disabled_collisions, \
    draw_pose, set_pose, set_joint_positions

from pychoreo.cartesian_planner.cartesian_process import CartesianTrajectory
from pychoreo.cartesian_planner.ladder_graph_interface import solve_ladder_graph_from_cartesian_processes
from pychoreo.utils.stream_utils import get_random_direction_generator, get_enumeration_pose_generator

import pychoreo_examples
from pychoreo_examples.extrusion.parsing import load_extrusion, create_elements_bodies
from pychoreo_examples.extrusion.visualization import set_extrusion_camera, draw_extrusion_sequence
from pychoreo_examples.extrusion.stream import extrusion_ee_pose_gen_fn
from pychoreo_examples.extrusion.utils import max_valence_extrusion_direction_routing, add_collision_fns_from_seq


import ikfast_kuka_kr6_r900

from compas.robots import RobotModel
from compas_fab.robots import Robot as RobotClass
from compas_fab.robots import RobotSemantics

def get_test_data(file_path):
    here = os.path.dirname(__file__)
    return os.path.join(here, 'test_data', file_path)

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
        ee_pose_gen_fn = extrusion_ee_pose_gen_fn(path_pts, random_dir_gen, interpolate_poses, pos_step_size=0.003)

        cart_traj = CartesianTrajectory(process_name=process_name,
            robot=robot, ik_joint_names=ik_joint_names,
            path_points=path_pts,
            ee_pose_gen_fn=ee_pose_gen_fn, sample_ik_fn=sample_ik_fn,
            element_identifier=element)

        ee_poses = cart_traj.sample_ee_poses()
        if viz_step:
            print('#{}'.format(element))
            for ee_p in ee_poses:
                ee_p = multiply(ee_p, tool_from_root)
                set_pose(ee_body, ee_p)
                if has_gui(): wait_for_user()

        with pytest.raises(NotImplementedError):
            # this should raise an not implemented error since we haven't specify the collision function yet
            ik_sols = cart_traj.get_ik_sols(ee_poses, check_collision=True)
            if all([not sol for sol in ik_sols]):
                conf = [0] * 6
                cart_traj.collision_fn(conf)

        ik_sols = cart_traj.get_ik_sols(ee_poses, check_collision=False)
        if viz_step:
            for jt_sol in ik_sols:
                for jts in jt_sol:
                    set_joint_positions(robot, ik_joints, jts)
                    if has_gui(): wait_for_user()

        cart_traj_dict[element] = cart_traj
    return cart_traj_dict


def test_extrusion_ladder_graph(viewer):
    # create robot and pb environment
    (robot_urdf, base_link_name, tool_root_link_name, ee_link_name, ik_joint_names, disabled_self_collision_link_names), \
        (workspace_urdf, workspace_robot_disabled_link_names) = get_robot_data()
    connect(use_gui=viewer)
    robot = load_pybullet(robot_urdf, fixed_base=True)
    workspace = load_pybullet(workspace_urdf, fixed_base=True)
    ik_fn = ikfast_kuka_kr6_r900.get_ik

    print(workspace_robot_disabled_link_names)

    # tool TCP from flange (tool0)
    root_link = link_from_name(robot, tool_root_link_name)
    # tool_body = clone_body(robot, links=tool_links, visual=False, collision=True)
    tool_link = link_from_name(robot, ee_link_name)
    tool_from_root = get_relative_pose(robot, root_link, tool_link)

    # load shape (get nodal positions)
    file_path = pychoreo_examples.get_data('assembly_instances/extrusion/simple_frame.json')
    elements, node_points, ground_nodes = load_extrusion(file_path)

    # create element bodies (collision geometries)
    with LockRenderer():
        draw_pose(unit_pose(), length=1.)
        element_bodies = dict(zip(elements,
            create_elements_bodies(node_points, elements, radius=0.001, shrink=0.008)))
        set_extrusion_camera(node_points)

    cart_process_dict = build_extrusion_cartesian_process(elements, node_points, robot, ik_fn, ik_joint_names,
        base_link_name, tool_from_root)

    # load precomputed sequence
    seq_file_path = get_test_data('simple_frame_solution_regression-z.json')
    with open(seq_file_path, 'r') as f:
        seq_data = json.loads(f.read())
    element_sequence = [tuple(e) for e in seq_data['plan']]

    reverse_flags = max_valence_extrusion_direction_routing(element_sequence, elements, ground_nodes)

    roll_disc = 20
    pitch_disc = 20
    domain_size = roll_disc * pitch_disc

    def get_ee_pose_map_fn(roll_disc, pitch_disc):
        def ee_pose_map_fn(id):
            j = id % roll_disc
            i = (id - j) / pitch_disc
            roll = -np.pi + i*(2*np.pi/roll_disc)
            pitch = -np.pi + j*(2*np.pi/pitch_disc)
            yaw = random.uniform(-np.pi, np.pi)
            return Pose(euler=Euler(roll=roll, pitch=pitch, yaw=yaw))
        return ee_pose_map_fn

    with WorldSaver():
        ee_body = load_extrusion_end_effector()
        ee_pose_map_fn = get_ee_pose_map_fn(roll_disc, pitch_disc)
        # building collision function based on the given sequence
        cart_process_seq, e_fmaps = add_collision_fns_from_seq(robot, ik_joint_names, cart_process_dict,
            element_sequence, element_bodies, domain_size, ee_pose_map_fn, ee_body,
            reverse_flags=reverse_flags, tool_from_root=tool_from_root,
            workspace_bodies=[workspace], ws_disabled_body_link_names=workspace_robot_disabled_link_names,
            disabled_self_collision_link_names=disabled_self_collision_link_names)

        assert isinstance(cart_process_seq, list)
        assert all([cart.element_identifier == e for cart, e in zip(cart_process_seq, element_sequence)])

        # just move the element bodies and ee_body away to clear the visualized scene
        set_pose(ee_body, unit_pose())
        for e_body in element_bodies.values(): set_pose(e_body, unit_pose())

        # draw the pruned EE direction set
        draw_extrusion_sequence(node_points, element_bodies, element_sequence, e_fmaps, ee_pose_map_fn=ee_pose_map_fn,
                                line_width=10, direction_len=0.005)

    tot_traj = solve_ladder_graph_from_cartesian_processes(cart_process_seq, verbose=True)

    # visualize plan
    ik_joints = joints_from_names(robot, ik_joint_names)
    for jts in tot_traj:
        set_joint_positions(robot, ik_joints, jts)
        if has_gui(): wait_for_user()

    if has_gui(): wait_for_user()
