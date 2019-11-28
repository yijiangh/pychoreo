import os
import pytest

from compas.robots import RobotModel
import compas_fab
from compas_fab.robots import Robot as RobotClass
from compas_fab.robots import RobotSemantics
import pychoreo_examples
from pybullet_planning import Pose

@pytest.fixture
def picknplace_problem_path():
    problem =  'dms_ws_tet_bars'
    here = os.path.dirname(__file__)
    pnp_file_path = os.path.join(here, '..', 'test_data', problem, 'json', problem + '.json')
    return pnp_file_path, problem + '_result_.json'

@pytest.fixture
def picknplace_robot_data():
    # * use the URDF files that are shipped with compas_fab
    # notice these files have ROS-package-based URDF paths
    robot_urdf = compas_fab.get('universal_robot/ur_description/urdf/ur5.urdf')
    robot_srdf = compas_fab.get('universal_robot/ur5_moveit_config/config/ur5.srdf')

    here = os.path.dirname(__file__)
    PNP_DIR = os.path.join(here, '..', 'test_data')
    workspace_urdf = os.path.join(PNP_DIR, 'dms_ws_tet_bars', 'urdf', 'dms_workspace.urdf')
    workspace_srdf = os.path.join(PNP_DIR, 'dms_ws_tet_bars', 'srdf', 'dms_workspace.srdf')

    move_group = None
    robot_model = RobotModel.from_urdf_file(robot_urdf)
    robot_semantics = RobotSemantics.from_srdf_file(robot_srdf, robot_model)
    robot = RobotClass(robot_model, semantics=robot_semantics)

    base_link_name = robot.get_base_link_name(group=move_group)
    # ee_link_name = robot.get_end_effector_link_name(group=move_group)
    ee_link_name = None
    ik_joint_names = robot.get_configurable_joint_names(group=move_group)
    disabled_self_collision_link_names = robot_semantics.get_disabled_collisions()
    tool_root_link_name = robot.get_end_effector_link_name(group=move_group) # TODO: should be derived from SRDF as well

    workspace_model = RobotModel.from_urdf_file(workspace_urdf)
    workspace_semantics = RobotSemantics.from_srdf_file(workspace_srdf, workspace_model)
    workspace_robot_disabled_link_names = workspace_semantics.get_disabled_collisions()
    workspace_robot_disabled_link_names = []

    return (robot_urdf, base_link_name, tool_root_link_name, ee_link_name, ik_joint_names, disabled_self_collision_link_names), \
           (workspace_urdf, workspace_robot_disabled_link_names)


@pytest.fixture
def picknplace_end_effector():
    here = os.path.dirname(__file__)
    return os.path.join(here, '..', 'test_data', 'dms_bar_gripper.obj')


@pytest.fixture
def picknplace_tcp_def():
    return Pose(point=[1e-3 * 96, 0, 0])
