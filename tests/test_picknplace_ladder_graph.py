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
from pybullet_planning import interpolate_poses, multiply, unit_pose, get_relative_pose
from pybullet_planning import interval_generator, sample_tool_ik
from pybullet_planning import Pose, Point, Euler, unit_pose
from pybullet_planning import joints_from_names, link_from_name, has_link, get_collision_fn, get_disabled_collisions, \
    draw_pose, set_pose, set_joint_positions, dump_world, create_obj, get_body_body_disabled_collisions, get_link_pose, \
    create_attachment, set_pose

from pychoreo.process_model.cartesian_process import CartesianProcess
from pychoreo.process_model.trajectory import Trajectory, MotionTrajectory
from pychoreo.utils.stream_utils import get_random_direction_generator, get_enumeration_pose_generator
from pychoreo.cartesian_planner.ladder_graph_interface import solve_ladder_graph_from_cartesian_process_list

import ikfast_ur5

def load_end_effector(ee_urdf_path):
    with HideOutput():
        ee = load_pybullet(ee_urdf_path)
    return ee

@pytest.mark.pnp
def test_picknplace_ladder_graph(viewer, picknplace_problem_path, picknplace_robot_data, picknplace_end_effector, picknplace_tcp_def):
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

    ik_fn = ikfast_ur5.get_ik

    # * create tool and tool TCP from flange (tool0) transformation
    root_link = link_from_name(robot, tool_root_link_name)
    # create end effector body
    ee_body = load_end_effector(picknplace_end_effector)
    # get current robot attach link pose
    ee_link_pose = get_link_pose(robot, root_link)
    # set the ee body's pose and create attachment
    set_pose(ee_body, ee_link_pose)
    ee_attach = create_attachment(robot, root_link, ee_body)
    # set up TCP transformation, just a renaming here
    root_from_tcp = picknplace_tcp_def
    if has_gui() :
        # draw_tcp pose
        draw_pose(multiply(ee_link_pose, root_from_tcp))

    # * get problem & pre-computed json file paths
    pkg_json_path = picknplace_problem_path

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

    # visualize goal pose
    if has_gui():
        viz_len = 0.003
        with WorldSaver():
            for element in elements.values():
                with LockRenderer():
                    handles = []
                    print(element)
                    for unit_geo in element.unit_geometries:
                        for pb_geo in unit_geo.pybullet_bodies:
                            print(pb_geo)
                            set_pose(pb_geo, random.choice(unit_geo.get_goal_frames(get_pb_pose=True)))
                        # visualize end effector pose
                        for initial_frame, pick_grasp in zip(unit_geo.get_initial_frames(get_pb_pose=True), unit_geo.pick_grasps):
                            handles.extend(draw_pose(initial_frame, length=0.01))
                            handles.extend(draw_pose(multiply(initial_frame, pick_grasp.get_object_from_approach_frame(get_pb_pose=True)), length=viz_len))
                            handles.extend(draw_pose(multiply(initial_frame, pick_grasp.get_object_from_attach_frame(get_pb_pose=True)), length=viz_len))
                            handles.extend(draw_pose(multiply(initial_frame, pick_grasp.get_object_from_retreat_frame(get_pb_pose=True)), length=viz_len))
                        for goal_frame, place_grasp in zip(unit_geo.get_goal_frames(get_pb_pose=True), unit_geo.place_grasps):
                            handles.extend(draw_pose(goal_frame, length=0.01))
                            handles.extend(draw_pose(multiply(goal_frame, place_grasp.get_object_from_approach_frame(get_pb_pose=True)), length=viz_len))
                            handles.extend(draw_pose(multiply(goal_frame, place_grasp.get_object_from_attach_frame(get_pb_pose=True)), length=viz_len))
                            handles.extend(draw_pose(multiply(goal_frame, place_grasp.get_object_from_retreat_frame(get_pb_pose=True)), length=viz_len))
                print('---------')
                wait_for_user()
                remove_handles(handles)

    # # * create cartesian processes without a sequence being given, with random pose generators
    # cart_process_dict = build_extrusion_cartesian_process(elements, node_points, robot, ik_fn, ik_joint_names,
    #     base_link_name, extrusion_end_effector, tool_from_root, viz_step=False)

    # * load precomputed sequence / use assigned sequence
    ## placeholder

    # * construct ignored body-body links for collision checking
    # in this case, including self-collision between links of the robot
    disabled_self_collisions = get_disabled_collisions(robot, disabled_self_collision_link_names)
    # and links between the robot and the workspace (e.g. robot_base_link to base_plate)
    extra_disabled_collisions = get_body_body_disabled_collisions(robot, workspace, workspace_robot_disabled_link_names)

    if has_gui() : wait_for_user()
