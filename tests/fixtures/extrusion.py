import os
import pytest

from compas.robots import RobotModel
from compas_fab.robots import Robot as RobotClass
from compas_fab.robots import RobotSemantics
import pychoreo_examples

@pytest.fixture
def extrusion_problem_path():
    # * extrusion problem here
    # problem = 'four-frame'
    # problem = 'extrusion_exp_L75.0'
    # problem = 'long_beam_test'
    # problem = 'extreme_beam_test'
    # problem = 'topopt-101_tiny'
    problem = 'topopt-205_long_beam_test'

    EXTRUSION_DIRECTORY = pychoreo_examples.get_data('assembly_instances/extrusion')
    EXTRUSION_FILENAMES = {
        'topopt-100': 'topopt-100_S1_03-14-2019_w_layer.json',
        'voronoi': 'voronoi_S1_03-14-2019_w_layer.json',
        'four-frame': 'four-frame.json',
        'simple_frame': 'simple_frame.json',
        'extrusion_exp_L75.0' : 'extrusion_exp_L75.0.json',
        'long_beam_test' : 'long_beam_test.json',
        'extreme_beam_test' : 'extreme_beam_test.json',
        'topopt-101_tiny' : 'topopt-101_tiny.json',
        'topopt-205_long_beam_test' : 'topopt-205_long_beam_test.json',
    }
    EXTRUSION_SEQ_FILENAMES = {
        'four-frame': 'four-frame_solution_regression-z.json',
        'simple_frame': 'simple_frame_solution_regression-z.json',
        'extrusion_exp_L75.0' : 'extrusion_exp_L75.0_solution_regression-z.json',
        'long_beam_test' : 'long_beam_test_solution_regression-z.json',
        'extreme_beam_test' : 'extreme_beam_test_solution_regression-z.json',
        'topopt-101_tiny' : 'topopt-101_tiny_solution_regression-dijkstra.json',
        'topopt-205_long_beam_test' : 'topopt-205_long_beam_test_solution_progression-dijkstra.json',
    }
    here = os.path.dirname(__file__)
    assert problem in EXTRUSION_FILENAMES and problem in EXTRUSION_SEQ_FILENAMES
    return os.path.join(EXTRUSION_DIRECTORY, EXTRUSION_FILENAMES[problem]), \
           os.path.join(here, '..', 'test_data', EXTRUSION_SEQ_FILENAMES[problem]), \
           os.path.join(here, '..', 'test_data', problem + '_pruned_ee_dir_result_.json'), \
           problem + '_result_.json'

@pytest.fixture
def extrusion_robot_data():
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

@pytest.fixture
def extrusion_end_effector():
    return pychoreo_examples.get_data('models/kuka_kr6_r900/urdf/extrusion_end_effector.urdf')
