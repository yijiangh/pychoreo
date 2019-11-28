import random
import numpy as np

from pybullet_planning import Pose, Point, Euler, unit_pose, invert, multiply, interpolate_poses_by_num_steps
from pybullet_planning import sample_tool_ik
from pybullet_planning import Attachment
from pybullet_planning import joints_from_names, link_from_name, has_link, get_collision_fn, get_disabled_collisions, \
    draw_pose, set_pose, set_joint_positions, dump_world, create_obj, get_body_body_disabled_collisions, get_link_pose, \
    create_attachment, set_pose, clone_body, has_gui, wait_for_user

from pychoreo.utils import is_any_empty
from pychoreo.process_model.cartesian_process import CartesianProcess, CartesianSubProcess
from pychoreo.process_model.gen_fn import CartesianPoseGenFn

from pychoreo_examples.picknplace.utils import flatten_dict_entries
from pychoreo_examples.picknplace.trajectory import PicknPlaceBufferTrajectory

def get_enumerate_picknplace_generator(unit_geo):
    for initial_frame, pick_grasp, goal_frame, place_grasp in zip(
                                         unit_geo.get_initial_frames(get_pb_pose=True), unit_geo.pick_grasps,\
                                         unit_geo.get_goal_frames(get_pb_pose=True),    unit_geo.place_grasps):
        pick_approach_landmarks = [multiply(initial_frame, pick_grasp.get_object_from_approach_frame(get_pb_pose=True)), \
                                   multiply(initial_frame, pick_grasp.get_object_from_attach_frame(get_pb_pose=True))]
        pick_retreat_landmarks = [multiply(initial_frame, pick_grasp.get_object_from_attach_frame(get_pb_pose=True)), \
                                  multiply(initial_frame, pick_grasp.get_object_from_retreat_frame(get_pb_pose=True))]

        place_approach_landmarks = [multiply(goal_frame, place_grasp.get_object_from_approach_frame(get_pb_pose=True)), \
                                    multiply(goal_frame, place_grasp.get_object_from_attach_frame(get_pb_pose=True))]
        place_retreat_landmarks = [multiply(goal_frame, place_grasp.get_object_from_attach_frame(get_pb_pose=True)), \
                                   multiply(goal_frame, place_grasp.get_object_from_retreat_frame(get_pb_pose=True))]

        ee_landmarks = [pick_approach_landmarks, pick_retreat_landmarks, place_approach_landmarks, place_retreat_landmarks]

        # from pybullet_planning import get_distance
        # for i, ee_p in enumerate(ee_landmarks):
        #     print('{} : distance {}'.format(i, get_distance(ee_p[0][0], ee_p[1][0])))
        yield ee_landmarks

def get_picknplace_ee_pose_compose_fn(ee_pose_interp_fn, **kwargs):
    def pose_compose_fn(ee_landmarks):
        process_path = []
        for end_pts_path in ee_landmarks:
            sub_path = []
            for i in range(len(end_pts_path)-1):
                sub_path.extend(ee_pose_interp_fn(end_pts_path[i], end_pts_path[i+1], **kwargs))
            process_path.append(sub_path)
        return process_path
    return pose_compose_fn

######################################

def build_picknplace_cartesian_process_seq(
        element_sequence, elements,
        robot, ik_joint_names, attach_link, sample_ik_fn,
        num_steps=5, ee_attachs=[],
        self_collisions=True, disabled_collisions={},
        obstacles=[], extra_disabled_collisions={},
        tool_from_root=None, viz_step=False, pick_from_same_rack=True):

    # load EE body, for debugging purpose
    ik_joints = joints_from_names(robot, ik_joint_names)

    # cloning all bodies in each unit_geometry to avoid syn problem
    element_bodies_pick = {}
    element_bodies_place = {}
    for e_id in element_sequence:
        # assume that each element is associated with only one unit geometry
        unit_geo = elements[e_id].unit_geometries[0]

        element_bodies_pick[e_id] = unit_geo.pybullet_bodies
        initial_pose = unit_geo.get_initial_frames(get_pb_pose=True)[0]
        for e_body in element_bodies_pick[e_id]:
            set_pose(e_body, initial_pose)

        # element_bodies_place[e_id] = [clone_body(body) for body in unit_geo.pybullet_bodies]
        element_bodies_place[e_id] = unit_geo.clone_pybullet_bodies()
        goal_pose = unit_geo.get_goal_frames(get_pb_pose=True)[0]
        for e_body in element_bodies_place[e_id]:
            set_pose(e_body, goal_pose)

    cart_process_seq = []
    assembled_element_obstacles = []
    for seq_id, e_id in enumerate(element_sequence):
        element = elements[e_id]
        unit_geo = elements[e_id].unit_geometries[0]

        grasp_enum_gen_fn = get_enumerate_picknplace_generator(unit_geo)
        pnp_compose_fn = get_picknplace_ee_pose_compose_fn(interpolate_poses_by_num_steps, num_steps=num_steps)
        pose_gen_fn = CartesianPoseGenFn(grasp_enum_gen_fn, pnp_compose_fn)

        # build collision_fn
        # ! element attachment cannot be built here since it's grasp sample-dependent
        # object_from_attach = unit_geo.pick_grasps[0].get_object_from_attach_frame(get_pb_pose=True)
        # # root_from_tool * tool-in-attach_from_object = root_from_object
        # pick_attach_from_object = multiply(invert(tool_from_root), invert(object_from_attach)) if tool_from_root else \
        #     invert(object_from_attach)
        # # notice that the shape in its goal pose (although there can be symmetric copy) should be the same
        element_attachs = [Attachment(robot, attach_link, None, e_body) for e_body in element_bodies_pick[e_id]]

        # approach 2 pick
        # ! end effector attachment is not modeled here
        # ! collision with the element being picked is not modeled here
        pick_approach_obstacles = obstacles + assembled_element_obstacles
        if not pick_from_same_rack:
            pick_approach_obstacles.extend(flatten_dict_entries(element_bodies_pick, range(0, seq_id)))
        # else:
        #     pick_approach_obstacles.extend(element_bodies_pick[seq_id]))
        pick_approach_collision_fn = get_collision_fn(robot, ik_joints, pick_approach_obstacles,
                                                      attachments=[], self_collisions=self_collisions,
                                                      disabled_collisions=disabled_collisions,
                                                      extra_disabled_collisions=extra_disabled_collisions,
                                                      custom_limits={})

        # pick 2 retreat
        # ! end effector attachment is not modeled here
        # ! collision with the element being picked is not modeled here
        pick_retreat_collision_fn = get_collision_fn(robot, ik_joints, pick_approach_obstacles,
                                                     attachments=[], self_collisions=self_collisions,
                                                     disabled_collisions=disabled_collisions,
                                                     extra_disabled_collisions=extra_disabled_collisions,
                                                     custom_limits={})

        # approach 2 place
        # ! collision with the unassembled element is not modeled here
        # ! no attachment is modelled now
        # TODO: ptwise collision check to exonerate touching between elements that are designed to be in contact
        # Now the workaround is shrinking the element's collision geometry
        place_approach_obstacles = obstacles + assembled_element_obstacles
        place_approach_collision_fn = get_collision_fn(robot, ik_joints,
                                                       place_approach_obstacles,
                                                       attachments=[], self_collisions=self_collisions,
                                                       disabled_collisions=disabled_collisions,
                                                       extra_disabled_collisions=extra_disabled_collisions,
                                                       custom_limits={})
        # place 2 retreat
        place_retreat_collision_fn = get_collision_fn(robot, ik_joints,
                                                      place_approach_obstacles,
                                                      attachments=[], self_collisions=self_collisions,
                                                      disabled_collisions=disabled_collisions,
                                                      extra_disabled_collisions=extra_disabled_collisions,
                                                      custom_limits={})

        # build sub-processes
        pnp_sub_procs = [CartesianSubProcess(sub_process_name='pick_approach', collision_fn=pick_approach_collision_fn),
                         CartesianSubProcess(sub_process_name='pick_retreat', collision_fn=pick_retreat_collision_fn),
                         CartesianSubProcess(sub_process_name='place_approach', collision_fn=place_approach_collision_fn),
                         CartesianSubProcess(sub_process_name='place_retreat', collision_fn=place_retreat_collision_fn)]

        # create trajectory containers for subprocesses
        pnp_sub_procs[0].trajectory = PicknPlaceBufferTrajectory(robot, ik_joints, None, ee_attachments=ee_attachs,
            tag='pick_approach', element_id=e_id, element_info=repr(element))
        pnp_sub_procs[1].trajectory = PicknPlaceBufferTrajectory(robot, ik_joints, None, ee_attachments=ee_attachs,
            tag='pick_retreat', element_id=e_id, element_info=repr(element))
        pnp_sub_procs[2].trajectory = PicknPlaceBufferTrajectory(robot, ik_joints, None, ee_attachments=ee_attachs,
            tag='place_approach', element_id=e_id, element_info=repr(element))
        pnp_sub_procs[3].trajectory = PicknPlaceBufferTrajectory(robot, ik_joints, None, ee_attachments=ee_attachs,
            tag='place_retreat', element_id=e_id, element_info=repr(element))

        process_name = 'picknplace-E#{}'.format(e_id)
        cart_process = CartesianProcess(process_name=process_name,
            robot=robot, ik_joint_names=ik_joint_names,
            sub_process_list=pnp_sub_procs,
            ee_pose_gen_fn=pose_gen_fn, sample_ik_fn=sample_ik_fn,
            element_identifier=e_id)

        cart_process_seq.append(cart_process)
        assembled_element_obstacles.extend(element_bodies_place[e_id])

        # / debug viz
        if viz_step:
            ee_poses = cart_process.sample_ee_poses()
            for sp_id, sp in enumerate(ee_poses):
                print('ID#{}:{} - sub process #{}'.format(e_id, element, sp_id))
                for ee_p in sp:
                    for ee_at in ee_attachs:
                        set_pose(ee_at.child, multiply(ee_p, tool_from_root))
                    if has_gui(): wait_for_user()

            ik_sols = cart_process.get_ik_sols(ee_poses, check_collision=True)
            for sp_id, sp_jt_sols in enumerate(ik_sols):
                for jt_sols in sp_jt_sols:
                    for jts in jt_sols:
                        set_joint_positions(robot, ik_joints, jts)
                        for ee_at in ee_attachs:
                            set_pose(ee_at.child, multiply(ee_p, tool_from_root))
                        if has_gui(): wait_for_user()
    return cart_process_seq
