import os
import json
import random
import numpy as np
import pytest
from itertools import chain

from pybullet_planning import load_pybullet, connect, wait_for_user, LockRenderer, has_gui, WorldSaver, HideOutput, \
    reset_simulation, disconnect, set_camera_pose
from pybullet_planning import interpolate_poses, multiply, unit_pose, get_relative_pose
from pybullet_planning import interval_generator, sample_tool_ik
from pybullet_planning import Pose, Point, Euler, unit_pose
from pybullet_planning import joints_from_names, link_from_name, has_link, get_collision_fn, get_disabled_collisions, \
    draw_pose, set_pose, set_joint_positions, dump_world, create_obj

from pychoreo.process_model.cartesian_process import CartesianProcess
from pychoreo.process_model.trajectory import Trajectory, MotionTrajectory
from pychoreo.utils.stream_utils import get_random_direction_generator, get_enumeration_pose_generator
from pychoreo.cartesian_planner.ladder_graph_interface import solve_ladder_graph_from_cartesian_processes
# from pychoreo.transition_planner.motion_planner_interface import solve_transition_between_cartesian_processes

import pychoreo_examples

import ikfast_ur5

import compas_fab
from compas.robots import RobotModel
from compas_fab.robots import Robot as RobotClass
from compas_fab.robots import RobotSemantics

@pytest.fixture
def problem():
    return 'dms_ws_tet_bars'

def get_problem_path(problem):
    here = os.path.dirname(__file__)
    PNP_DIR = os.path.join(here, 'test_data')
    return os.path.join(PNP_DIR, problem)

def get_robot_data():
    # * use the URDF files that are shipped with compas_fab
    # notice these files have ROS-package-based URDF paths
    robot_urdf = compas_fab.get('universal_robot/ur_description/urdf/ur5.urdf')
    robot_srdf = compas_fab.get('universal_robot/ur5_moveit_config/config/ur5.srdf')

    workspace_urdf = None
    workspace_srdf = None

    move_group = None
    robot_model = RobotModel.from_urdf_file(robot_urdf)
    robot_semantics = RobotSemantics.from_srdf_file(robot_srdf, robot_model)
    robot = RobotClass(robot_model, semantics=robot_semantics)

    base_link_name = robot.get_base_link_name(group=move_group)
    # ee_link_name = robot.get_end_effector_link_name(group=move_group)
    ee_link_name = None
    ik_joint_names = robot.get_configurable_joint_names(group=move_group)
    disabled_self_collision_link_names = robot_semantics.get_disabled_collisions()
    tool_root_link_name = 'tool0' # TODO: should be derived from SRDF as well

    # workspace_model = RobotModel.from_urdf_file(workspace_urdf)
    # workspace_semantics = RobotSemantics.from_srdf_file(workspace_srdf, workspace_model)
    # workspace_robot_disabled_link_names = workspace_semantics.get_disabled_collisions()
    workspace_robot_disabled_link_names = []

    return (robot_urdf, base_link_name, tool_root_link_name, ee_link_name, ik_joint_names, disabled_self_collision_link_names), \
           (workspace_urdf, workspace_robot_disabled_link_names)

def load_extrusion_end_effector():
    here = os.path.dirname(__file__)
    with HideOutput():
        ee = create_obj(os.path.join(here, 'test_data', 'dms_bar_gripper.obj'))
    return ee

@pytest.mark.pnp
def test_picknplace_ladder_graph(problem, viewer):
    # * create robot and pb environment
    (robot_urdf, base_link_name, tool_root_link_name, ee_link_name, ik_joint_names, disabled_self_collision_link_names), \
    (workspace_urdf, workspace_robot_disabled_link_names) = get_robot_data()
    connect(use_gui=viewer)

    # * adjust camera pose (optional)
    camera_base_pt = (0,0,0)
    camera_pt = np.array(camera_base_pt) + np.array([1, 0, 0.5])
    set_camera_pose(tuple(camera_pt), camera_base_pt)

    with HideOutput():
        # * pybullet can handle ROS-package path URDF automatically now (ver 2.5.7)!
        robot = load_pybullet(robot_urdf, fixed_base=True)

    # * set robot idle configuration
    ik_joints = joints_from_names(robot, ik_joint_names)
    robot_start_conf = [0,-1.65715,1.71108,-1.62348,0,0]
    set_joint_positions(robot, ik_joints, robot_start_conf)

    ik_fn = ikfast_ur5.get_ik

    # * create tool and tool TCP from flange (tool0) transformation
    root_link = link_from_name(robot, tool_root_link_name)
    if ee_link_name:
        tool_link = link_from_name(robot, ee_link_name)
        tool_from_root = get_relative_pose(robot, root_link, tool_link)
    else:
        # TODO: create two new links: ee_base_link, ee_tcp_frame
        # TODO: create two joints: ee_base_to_tcp, ee_to_robot
        # attach ee_link (w/ collision geometry) to tool0
        # attach ee_tcp_link (no geometry) to tool0
        ee_body = load_extrusion_end_effector()
        ee_link_name = 'ee_tcp_link'

    dump_world()

    # * get problem & pre-computed json file paths
    pkg_path = get_problem_path(problem)

    if has_gui() : wait_for_user()
