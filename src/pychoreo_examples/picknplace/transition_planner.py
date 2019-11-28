from itertools import product
from pybullet_planning import BASE_LINK
from pybullet_planning import set_joint_positions, set_pose
from pybullet_planning import plan_joint_motion, get_collision_fn

from pychoreo.process_model.trajectory import MotionTrajectory

def solve_transition_between_picknplace_processes(trajs, elements, initial_conf,
                                                  obstacles=[], return2idle=True, self_collisions=True,
                                                  disabled_collisions=set(), extra_disabled_collisions=set(),
                                                  custom_limits={}, **kwargs):
    print('*' * 10)
    print('transition planning starts.')
    built_obstacles = []
    for element in elements.values():
        unit_geo = element.unit_geometries[0]
        for e_body in unit_geo.pybullet_bodies:
            set_pose(e_body, unit_geo.get_initial_frames(get_pb_pose=True)[0])
            built_obstacles.append(e_body)

    trans_traj = []
    for seq_id, cart_traj in enumerate(trajs):
        cp_trans_trajs = []
        sp_ids = [0,2] if seq_id < len(trajs)-1 or not return2idle else [0,2,4]
        for sp_id in sp_ids:
            if sp_id == 0:
                tag = 'place2pick'
                sp_traj = cart_traj[sp_id]
                print('transition seq #{} - to {}'.format(seq_id, sp_traj.tag))
                if seq_id != 0:
                    tr_start_conf = trajs[seq_id-1][-1].traj_path[-1]
                else:
                    tag = 'initial2pick'
                    tr_start_conf = initial_conf
                tr_end_conf = trajs[seq_id][0].traj_path[0]
            elif sp_id == 2:
                tag = 'pick2place'
                sp_traj = cart_traj[sp_id]
                sp_traj = cart_traj[sp_id]
                print('transition seq #{} - to {}'.format(seq_id, sp_traj.tag))
                tr_start_conf = trajs[seq_id][1].traj_path[-1]
                tr_end_conf = trajs[seq_id][2].traj_path[0]
            elif sp_id == 4:
                tag = 'return2idle'
                print('plan for returning to idle position.')
                sp_traj = cart_traj[-1]
                tr_start_conf = sp_traj.traj_path[-1]
                tr_end_conf = initial_conf

            robot = sp_traj.robot
            ik_joints = sp_traj.joints

            ee_attachments = sp_traj.ee_attachments
            attachments = sp_traj.attachments
            if sp_id == 2:
                ee_element_disabled = product([(e_at.child, BASE_LINK) for e_at in ee_attachments], \
                                              [(at.child, BASE_LINK)   for at in attachments])
                sp_extra_disabled_collisions = extra_disabled_collisions.union(ee_element_disabled)
            else:
                sp_extra_disabled_collisions = extra_disabled_collisions

            set_joint_positions(robot, ik_joints, tr_start_conf)
            tr_path = plan_joint_motion(robot, ik_joints, tr_end_conf,
                                        obstacles=obstacles + built_obstacles, attachments=ee_attachments + attachments,
                                        self_collisions=self_collisions, disabled_collisions=disabled_collisions,
                                        extra_disabled_collisions=sp_extra_disabled_collisions,
                                        custom_limits=custom_limits, **kwargs)
            if not tr_path:
                print('subprocess {} cannot find transition path'.format(sp_traj))
                print('Diagnosis...')

                cfn = get_collision_fn(robot, ik_joints,
                                       obstacles=obstacles + built_obstacles, attachments=ee_attachments + attachments,
                                       self_collisions=self_collisions, disabled_collisions=disabled_collisions,
                                       extra_disabled_collisions=sp_extra_disabled_collisions, custom_limits=custom_limits)

                print('start pose:')
                print('in collision? ', cfn(tr_start_conf, diagnosis=True))

                print('end pose:')
                print('in collision? ', cfn(tr_end_conf, diagnosis=True))
                print('------------')
            cp_trans_trajs.append(MotionTrajectory(robot, ik_joints, tr_path,
                ee_attachments=ee_attachments, attachments=attachments,
                tag=tag, element_id=sp_traj.element_id))

            if sp_id == 2:
                # pick2place, go to the detach conf and leave the element there
                detach_conf = trajs[seq_id][2].traj_path[-1]
                set_joint_positions(robot, ik_joints, detach_conf)
                for at in attachments: at.assign()

        trans_traj.append(cp_trans_trajs)
    return trans_traj
