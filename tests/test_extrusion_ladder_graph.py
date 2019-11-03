import os
import json
import numpy as np
import pytest

from pybullet_planning import load_pybullet, connect, wait_for_user
from pybullet_planning import interpolate_poses, multiply
from pybullet_planning import interval_generator, sample_tool_ik
from pybullet_planning import joints_from_names
from pybullet_planning import Pose, Point, Euler, HideOutput

from pychoreo.cartesian_planner.cartesian_process import CartesianTrajectory
import pychoreo_examples
from pychoreo_examples.extrusion import load_extrusion

from compas.robots import RobotModel
from compas_fab.robots import Robot as RobotClass
from compas_fab.robots import RobotSemantics

def get_test_data(file_path):
    here = os.path.dirname(__file__)
    return os.path.join(here, 'test_data', file_path)

def get_robot_data():
    URDF_PATH = 'models/kuka_kr6_r900/urdf/kuka_kr6_r900_extrusion_mit_3-412.urdf'
    robot_urdf = pychoreo_examples.get_data(URDF_PATH)

    SRDF_PATH = 'models/kuka_kr6_r900/srdf/kuka_kr6_r900_extrusion_mit_3-412.srdf'
    robot_srdf = pychoreo_examples.get_data(SRDF_PATH)

    move_group = 'manipulator_ee'

    model = RobotModel.from_urdf_file(robot_urdf)
    semantics = RobotSemantics.from_srdf_file(robot_srdf, model)
    robot = RobotClass(model, semantics=semantics)

    base_link_name = robot.get_base_link_name(group=move_group)
    ee_link_name = robot.get_end_effector_link_name(group=move_group)
    ik_joint_names = robot.get_configurable_joint_names(group=move_group)
    disabled_link_names = semantics.get_disabled_collisions()

    return robot_urdf, base_link_name, ee_link_name, ik_joint_names, disabled_link_names

def load_extrusion_end_effector():
    with HideOutput():
        ee = load_pybullet(pychoreo_examples.get_data('models/kuka_kr6_r900/urdf/extrusion_end_effector.urdf'))
    return ee

def build_extrusion_cartesian_process(robot, ik_fn, ik_joint_names, base_link_name, disabled_links):
    def get_direction_generator(**kwargs):
        lower = [-np.pi, -np.pi]
        upper = [+np.pi, +np.pi]
        for [roll, pitch] in interval_generator(lower, upper, **kwargs):
            pose = Pose(euler=Euler(roll=roll, pitch=pitch))
            yield pose

    def get_extrusion_ee_pose_gen_fn(start_pt, end_pt):
        direction_gen_fn = get_direction_generator(use_halton=False)
        def pose_gen():
            ee_orient = next(direction_gen_fn)
            yield [multiply(Pose(point=Point(*pt)), ee_orient) for pt in [start_pt, end_pt]]
        return pose_gen

    def get_sample_ik_fn(robot, ik_fn, ik_joint_names, base_link_name):
        def sample_ik_fn(world_from_tcp):
            return sample_tool_ik(ik_fn, robot, ik_joint_names, base_link_name, world_from_tcp, get_all=True)
        return sample_ik_fn

    # load shape (get nodal positions)
    file_path = pychoreo_examples.get_data('assembly_instances/extrusion/simple_frame.json')
    elements, node_points, ground_nodes = load_extrusion(file_path)

    # create element bodies (collision geometries)

    # load precomputed sequence
    seq_file_path = get_test_data('simple_frame_solution_regression-z.json')
    with open(seq_file_path, 'r') as f:
        seq_data = json.loads(f.read())
    seq = seq_data['plan']

    # create cartesian trajectory
    sample_ik_fn = get_sample_ik_fn(robot, ik_fn, ik_joint_names, base_link_name)
    cart_traj_seq = []
    for seq_id, element in enumerate(seq):
        process_name = 'extrusion#{}-E{}'.format(seq_id, element)
        n1, n2 = element
        # TODO: pruned set enumeration generator
        ee_pose_gen_fn = get_extrusion_ee_pose_gen_fn(node_points[n1], node_points[n2])

        built_obstacles = []
        disabled_collisions = disabled_collisions

        ik_joints = joints_from_names(robot, ik_joint_names)
        collision_fn = get_collision_fn(robot, ik_joints, built_obstacles,
                                        attachments=[], self_collisions=True,
                                        disabled_collisions=disabled_collisions,
                                        custom_limits={})
        cart_traj = CartesianTrajectory(process_name, ee_pose_gen_fn, interpolate_poses, sample_ik_fn, collision_fn)

    # return extrusion_trajs

def test_extrusion_ladder_graph(viewer):
    # create robot and pb environment
    robot_urdf, base_link_name, ee_link_name, ik_joint_names, disabled_link_names = get_robot_data()
    connect(use_gui=viewer)
    robot = load_pybullet(robot_urdf, fixed_base=True)

    # build_extrusion_cartesian_process()
    wait_for_user()

def test_extrusion_forward_direction_pruning():
    pass
