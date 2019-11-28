import time
import numpy as np
import colorsys
from pybullet_planning import LockRenderer, set_camera_pose, add_line, add_text, wait_for_user, connect, load_pybullet, \
    reset_simulation, disconnect, wait_for_duration, HideOutput, draw_pose
from pybullet_planning import joints_from_names, set_joint_positions
from pybullet_planning import get_link_pose, link_from_name
from pybullet_planning import unit_pose, multiply, tform_point, point_from_pose
from pybullet_planning import Pose, Point, Euler, INF

from pychoreo.process_model.trajectory import MotionTrajectory
from pychoreo_examples.extrusion.trajectory import PrintTrajectory, PrintBufferTrajectory
from pychoreo_examples.extrusion.utils import is_ground

##################################################

def draw_element(node_points, element, color=(1, 0, 0)):
    n1, n2 = element
    p1 = node_points[n1]
    p2 = node_points[n2]
    return add_line(p1, p2, color=color[:3])


def sample_colors(num, lower=0.0, upper=0.75): # for now wrap around
    return [colorsys.hsv_to_rgb(h, s=1, v=1) for h in np.linspace(lower, upper, num, endpoint=True)]


def draw_ordered(elements, node_points):
    #colors = spaced_colors(len(elements))
    colors = sample_colors(len(elements))
    handles = []
    for element, color in zip(elements, colors):
        handles.append(draw_element(node_points, element, color=color))
    return handles

##################################################

def set_extrusion_camera(node_points):
    centroid = np.average(node_points, axis=0)
    camera_offset = 0.25 * np.array([1, -1, 1])
    set_camera_pose(camera_point=centroid + camera_offset, target_point=centroid)

##################################################

def draw_extrusion_sequence(node_points, element_bodies, element_sequence, seq_poses=None, ee_pose_map_fn=lambda id : unit_pose,
                           line_width=10, time_step=INF, direction_len=0.005, extrusion_tags=[]):
    assert len(element_bodies) == len(element_sequence)
    if seq_poses:
        assert len(seq_poses) == len(element_sequence)
    handles = []

    for seq_id, element in enumerate(element_sequence):
        print('visualizing EE directions for seq #{}'.format(seq_id))
        n1, n2 = element
        p1, p2 = (node_points[n1], node_points[n2])
        e_mid = (np.array(p1) + np.array(p2)) / 2

        seq_ratio = float(seq_id)/(len(element_sequence)-1)
        color = np.array([0, 0, 1])*(1-seq_ratio) + np.array([1,0,0])*seq_ratio
        handles.append(add_line(p1, p2, color=tuple(color), width=line_width))
        element_tag = str(seq_id)
        if len(extrusion_tags) == len(element_sequence):
            element_tag += '-' + extrusion_tags[seq_id]
        handles.append(add_text(element_tag, position=e_mid))

        if seq_poses is not None:
            with LockRenderer():
                for flag_id, feasible_flag in enumerate(seq_poses[element]):
                    if feasible_flag:
                        ee_pose = ee_pose_map_fn(flag_id)
                        fmap_pose = multiply(Pose(point=e_mid), ee_pose)
                        origin_world = tform_point(fmap_pose, np.zeros(3))
                        axis = np.zeros(3)
                        axis[2] = 1
                        axis_world = tform_point(fmap_pose, direction_len*axis)
                        handles.append(add_line(origin_world, axis_world, color=axis))
                        # handles.append(draw_pose(cmap_pose, direction_len))
        if time_step < INF:
            time.sleep(time_step)
        else:
            wait_for_user()

##################################################

def display_trajectories(robot_urdf, ik_joint_names, ee_link_name, node_points, ground_nodes, trajectories,
                         workspace_urdf=None, animate=True, cart_time_step=0.02, tr_time_step=0.05):
    if trajectories is None:
        return
    connect(use_gui=True)
    set_extrusion_camera(node_points)
    with HideOutput():
        robot = load_pybullet(robot_urdf, fixed_base=True)
        if workspace_urdf: workspace = load_pybullet(workspace_urdf, fixed_base=True)
    ik_joints = joints_from_names(robot, ik_joint_names)

    if not animate:
        planned_elements = [traj.element for traj in trajectories]
        with LockRenderer():
            draw_ordered(planned_elements, node_points)
        print('Planned sequence visualized.')
        wait_for_user()
        disconnect()
        return

    print('Ready to start simulation of the planned Trajectory.')
    # wait_for_user()
    #element_bodies = dict(zip(elements, create_elements(node_points, elements)))
    #for body in element_bodies.values():
    #    set_color(body, (1, 0, 0, 0))
    connected_nodes = set(ground_nodes)
    for cp_id, cp_trajs in enumerate(trajectories):
        #wait_for_user()
        #set_color(element_bodies[element], (1, 0, 0, 1))
        last_point = None
        handles = []
        for trajectory in cp_trajs:
            for conf in trajectory.traj_path:
                set_joint_positions(robot, ik_joints, conf)
                # * tracing TCP to represent extrusion
                if isinstance(trajectory, PrintTrajectory):
                    ee_pose = get_link_pose(robot, link_from_name(robot, ee_link_name))
                    # draw_pose(ee_pose, length=0.005)
                    current_point = point_from_pose(ee_pose)
                    if last_point is not None:
                        color = (0, 0, 1) if is_ground(trajectory.element, ground_nodes) else (1, 0, 0)
                        handles.append(add_line(last_point, current_point, color=color))
                    last_point = current_point
                    if cart_time_step is None:
                        wait_for_user()
                    else:
                        # ! this seems to have a bug on windows
                        # wait_for_duration(time_step)
                        time.sleep(cart_time_step)
                elif isinstance(trajectory, PrintBufferTrajectory):
                    if cart_time_step is None:
                        wait_for_user()
                    else:
                        # ! this seems to have a bug on windows
                        # wait_for_duration(time_step)
                        time.sleep(cart_time_step)
                elif isinstance(trajectory, MotionTrajectory):
                    if tr_time_step is None:
                        wait_for_user()
                    else:
                        # ! this seems to have a bug on windows
                        # wait_for_duration(time_step)
                        time.sleep(tr_time_step)
                else:
                    raise ValueError('trajectory type not acceptable!')

            # * sanity check on connectness
            if isinstance(trajectory, PrintTrajectory):
                is_connected = (trajectory.n1 in connected_nodes) # and (trajectory.n2 in connected_nodes)
                print('{}) {:9} | Tag: {} | Connected: {} | Ground: {} | Length: {}'.format(
                    cp_id, str(trajectory), trajectory.tag, is_connected, is_ground(trajectory.directed_element, ground_nodes), len(trajectory.traj_path)))
                if not is_connected:
                    print('Warning: this element is not connected to existing partial structure!')
                    wait_for_user()
                connected_nodes.add(trajectory.n2)

    wait_for_user()
    reset_simulation()
    disconnect()
