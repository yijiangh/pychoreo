import json
import time
import numpy as np
import colorsys
from pybullet_planning import LockRenderer, set_camera_pose, add_line, add_text, wait_for_user, connect, load_pybullet, \
    reset_simulation, disconnect, wait_for_duration, HideOutput, draw_pose, set_pose, create_attachment
from pybullet_planning import joints_from_names, set_joint_positions
from pybullet_planning import get_link_pose, link_from_name
from pybullet_planning import unit_pose, multiply, tform_point, point_from_pose
from pybullet_planning import Pose, Point, Euler, INF, Attachment

from compas_fab.assembly.datastructures import Assembly, UnitGeometry, Grasp

from pychoreo.process_model.trajectory import MotionTrajectory
from pychoreo_examples.picknplace.trajectory import PicknPlaceBufferTrajectory

def display_picknplace_trajectories(robot_urdf, ik_joint_names,
                                    pkg_json_path, trajectories, tool_root_link_name,
                                    ee_urdf=None, workspace_urdf=None, animate=True, cart_time_step=0.02, tr_time_step=0.05):
    if trajectories is None:
        return
    connect(use_gui=True)

    # * adjust camera pose (optional)
    camera_base_pt = (0,0,0)
    camera_pt = np.array(camera_base_pt) + np.array([1, 0, 0.5])
    set_camera_pose(tuple(camera_pt), camera_base_pt)

    with HideOutput():
        robot = load_pybullet(robot_urdf, fixed_base=True)
        root_link = link_from_name(robot, tool_root_link_name)
        ik_joints = joints_from_names(robot, ik_joint_names)

        if workspace_urdf: workspace = load_pybullet(workspace_urdf, fixed_base=True)
        if ee_urdf:
            ee_body = load_pybullet(ee_urdf)

    # * load shape & collision data
    with open(pkg_json_path, 'r') as f:
        json_data = json.loads(f.read())
    assembly = Assembly.from_package(json_data)
    elements = assembly.elements
    for element in elements.values():
        for unit_geo in element.unit_geometries:
            unit_geo.rescale(1e-3)

    if not animate:
        # planned_elements = [traj.element for traj in trajectories]
        # with LockRenderer():
        #     draw_ordered(planned_elements, node_points)
        print('Planned sequence visualized.')
        wait_for_user()
        disconnect()
        return

    for element in elements.values():
        unit_geo = element.unit_geometries[0]
        for e_body in unit_geo.pybullet_bodies:
            set_pose(e_body, unit_geo.get_initial_frames(get_pb_pose=True)[0])

    print('Ready to start simulation of the planned Trajectory.')
    for cp_id, cp_trajs in enumerate(trajectories):
        element_attachs = []
        for trajectory in cp_trajs:
            for conf_id, conf in enumerate(trajectory.traj_path):
                set_joint_positions(trajectory.robot, trajectory.joints, conf)

                if isinstance(trajectory, PicknPlaceBufferTrajectory):
                    for attach in trajectory.attachments:
                        if isinstance(attach.child, str):
                            if attach.child == 'body2':
                                attach.child = ee_body
                            else:
                                # TODO: temporal workaround...
                                attach.child = elements[trajectory.element_id].unit_geometries[0].pybullet_bodies[0]
                        attach.assign()

                    if cart_time_step is None:
                        wait_for_user()
                    else:
                        # wait_for_duration(time_step)
                        time.sleep(cart_time_step)
                elif isinstance(trajectory, MotionTrajectory):
                    if tr_time_step is None:
                        wait_for_user()
                    else:
                        time.sleep(tr_time_step)
                else:
                    raise ValueError('trajectory type not acceptable!')

    wait_for_user()
    reset_simulation()
    disconnect()
