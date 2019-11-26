import random

from pybullet_planning import Pose, Point, Euler, unit_pose, invert, multiply, interpolate_poses
from pybullet_planning import sample_tool_ik
from pybullet_planning import Attachment
from pybullet_planning import joints_from_names, link_from_name, has_link, get_collision_fn, get_disabled_collisions, \
    draw_pose, set_pose, set_joint_positions, dump_world, create_obj, get_body_body_disabled_collisions, get_link_pose, \
    create_attachment, set_pose

from pychoreo.utils import is_any_empty
from pychoreo.process_model.cartesian_process import CartesianProcess, CartesianSubProcess
from pychoreo.process_model.gen_fn import CartesianPoseGenFn

from pychoreo_examples.picknplace.stream import picknplace_ee_pose_gen_fn

def build_picknplace_cartesian_process_seq(elements, element_sequence, robot, ik_fn, ik_joint_names, base_link_name, ee_attachs,
        self_collisions=True, disabled_collisions={},
        obstacles=None, extra_disabled_collisions={},
        tool_from_root=None, viz_step=False):
    def get_sample_ik_fn(robot, ik_fn, ik_joint_names, base_link_name, tool_from_root=None):
        def sample_ik_fn(world_from_tcp):
            if tool_from_root:
                world_from_tcp = multiply(world_from_tcp, tool_from_root)
            return sample_tool_ik(ik_fn, robot, ik_joint_names, base_link_name, world_from_tcp, get_all=True, sampled=[0])
        return sample_ik_fn

    # ik generation function stays the same for all cartesian processes
    sample_ik_fn = get_sample_ik_fn(robot, ik_fn, ik_joint_names, base_link_name, tool_from_root)

    # load EE body, for debugging purpose
    ik_joints = joints_from_names(robot, ik_joint_names)

    cart_process_seq = []
    for e_id in element_sequence:
        element = elements[e_id]
        process_name = 'picknplace-E#{}'.format(e_id)

        # assume that each element is associated with only one unit geometry
        unit_geo = element.unit_geometries[0]
        ee_pose_gen_fn = CartesianPoseGenFn(
            picknplace_ee_pose_gen_fn(unit_geo, interpolate_poses, pos_step_size=0.003))

        # build sub-processes
        pnp_sub_procs = [CartesianSubProcess(sub_process_name='pick_approach', collision_fn=pick_approach_collision_fn),
                         CartesianSubProcess(sub_process_name='pick_retreat', collision_fn=pick_retreat_collision_fn),
                         CartesianSubProcess(sub_process_name='place_approach', collision_fn=place_approach_collision_fn),
                         CartesianSubProcess(sub_process_name='place_retreat', collision_fn=place_retreat_collision_fn)]

        cart_process = CartesianProcess(process_name=process_name,
            robot=robot, ik_joint_names=ik_joint_names,
            sub_process_list=pnp_sub_procs,
            ee_pose_gen_fn=ee_pose_gen_fn, sample_ik_fn=sample_ik_fn,
            element_identifier=e_id)

        # build collision_fn
        for ee_poses in cart_process.exhaust_iter():
            ik_sols = cart_process.get_ik_sols(ee_poses, check_collision=False)
            if not is_any_empty(ik_sols):
                break
        assert not is_any_empty(ik_sols), 'no ik solution found even without considering collisions.'

        attach_from_object = multiply(invert(tool_from_root), invert(pick_grasp.object_from_attach_pb_pose)) if tool_from_root else \
            invert(pick_grasp.object_from_attach_pb_pose)

        # the place attach configuration
        set_joint_positions(robot, ik_joints, ik_sols[2][-1])

        # the initial frame should correspond to the iter but just do it for now...
        # ! notice that the shape in its goal pose (although there can be symmetric copy) should be the same
        initial_pose = random.choice(unit_geo.get_goal_frames(get_pb_pose=True))
        for e_body in unit_geo.pybullet_bodies:
            set_pose(e_body, unit_geo.initial_pose)

        attachs = [Attachment(robot, tool_link, attach_from_object, e_body) for e_body in unit_geo.pybullet_bodies]
        if ee_attachs:
            attachs.extend(ee_attachs)

        # ee_poses = cart_process.sample_ee_poses()
        # for sp_id, sp in enumerate(ee_poses):
        #     print('{} - sub process #{}'.format(element, sp_id))
        #     for ee_p in sp:
        #         ee_p = multiply(ee_p, tool_from_root)
        #         for ee_at in ee_attachs:
        #             set_pose(ee_at.child, ee_p)
        #         if has_gui(): wait_for_user()

        # ik_sols = cart_process.get_ik_sols(ee_poses, check_collision=False)
        # for sp_id, sp_jt_sols in enumerate(ik_sols):
        #     for jt_sols in sp_jt_sols:
        #         for jts in jt_sols:
        #             set_joint_positions(robot, ik_joints, jts)
        #             if has_gui(): wait_for_user()

        cart_process_seq.append(cart_process)
    return cart_process_seq
